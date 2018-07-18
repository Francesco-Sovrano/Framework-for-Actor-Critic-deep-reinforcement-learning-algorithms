# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import tensorflow as tf
import numpy as np
from model.network import *
from model.experience_buffer import Buffer

import options
flags = options.get()

class BasicManager(object):
	def __init__(self, session, device, id, action_size, state_shape, concat_size=0, global_network=None):
		self.session = session
		self.id = id
		self.device = device
		self.global_network = global_network
		self.set_model_size()
	# Gradient optimizer and clip range
		if not self.is_global_network():
			self.clip = self.global_network.clip
		else:
			self.initialize_gradient_optimizer()
	# Build agents
		self.model_list = []
		self._model_usage_list = deque()
		self.build_agents(state_shape=state_shape, action_size=action_size, concat_size=concat_size)
	# Build experience buffer
		if flags.replay_ratio > 0:
			self.experience_buffer = Buffer(size=flags.replay_size)
	# Bind optimizer to global
		if not self.is_global_network():
			self.bind_to_global(self.global_network)
			
	def is_global_network(self):
		return self.global_network is None
			
	def set_model_size(self):
		self.model_size = 1
			
	def build_agents(self, state_shape, action_size, concat_size):
		agent=eval(flags.network + "_Network")(
			session=self.session,
			id="{0}_{1}".format(self.id, 0),
			device=self.device,
			state_shape=state_shape,
			policy_size=action_size,
			concat_size=concat_size,
			entropy_beta=flags.entropy_beta,
			clip=self.clip[0],
			predict_reward=flags.predict_reward
		)
		self.model_list.append(agent)
			
	def sync(self):
		assert not self.is_global_network(), 'you are trying to sync the global network with itself'
		for i in range(self.model_size):
			agent = self.model_list[i]
			sync = self.sync_list[i]
			agent.sync(sync)
			
	def initialize_gradient_optimizer(self):
		self.global_step = []
		self.learning_rate = []
		self.clip = []
		self.gradient_optimizer = []
		for i in range(self.model_size):
		# global step
			self.global_step.append( tf.Variable(0, trainable=False) )
		# learning rate
			self.learning_rate.append( eval('tf.train.'+flags.alpha_annealing_function)(learning_rate=flags.alpha, global_step=self.global_step[i], decay_steps=flags.alpha_decay_steps, decay_rate=flags.alpha_decay_rate) if flags.alpha_decay else flags.alpha )
		# clip
			self.clip.append( eval('tf.train.'+flags.clip_annealing_function)(learning_rate=flags.clip, global_step=self.global_step[i], decay_steps=flags.clip_decay_steps, decay_rate=flags.clip_decay_rate) if flags.clip_decay else flags.clip )
		# gradient optimizer
			self.gradient_optimizer.append( eval('tf.train.'+flags.optimizer+'Optimizer')(learning_rate=self.learning_rate[i], use_locking=True) )
			
	def bind_to_global(self, global_network):
		self.sync_list = []
		for i in range(self.model_size):
			local_agent = self.get_model(i)
			global_agent = global_network.get_model(i)
			local_agent.prepare_loss()
			local_agent.minimize_local(optimizer=global_network.gradient_optimizer[i], global_step=global_network.global_step[i], global_var_list=global_agent.get_vars())
			self.sync_list.append(local_agent.bind_sync(global_agent)) # for synching local network with global one

	def get_model(self, id):
		return self.model_list[id]
		
	def get_statistics(self):
		if self.model_size == 1:
			return {}
		stats = {}
		total_usage = 0
		usage_matrix = {}
		for u in self._model_usage_list:
			if not (u in usage_matrix):
				usage_matrix[u] = 0
			usage_matrix[u] += 1
			total_usage += 1
		for i in range(self.model_size):
			stats["model_{0}".format(i)] = 0
		for key, value in usage_matrix.items():
			stats["model_{0}".format(key)] = value/total_usage if total_usage != 0 else 0
		return stats
		
	def add_to_statistics(self, id):
		self._model_usage_list.append(id)
		if len(self._model_usage_list) > flags.match_count_for_evaluation:
			self._model_usage_list.popleft() # remove old statistics
		
	def get_vars(self):
		vars = []
		for agent in self.model_list:
			vars += agent.get_vars()
		return vars
		
	def reset(self):
		self.agent_id = 0
		for agent in self.model_list:
			agent.reset()
			
	def initialize_new_batch(self):
		self.batch = {}
		self.batch["states"] = []
		self.batch["concats"] = []
		self.batch["actions"] = []
		self.batch["rewards"] = []
		self.batch["values"] = []
		self.batch["policies"] = []
		self.batch["lstm_states"] = []
		self.batch["discounted_cumulative_rewards"] = []
		self.batch["generalized_advantage_estimators"] = []
		for i in range(self.model_size):
			for key in self.batch:
				self.batch[key].append(deque())
		
		self.batch["agent-position_list"] = []
		self.batch["total_reward"] = 0
		self.batch["size"] = 0
		self.batch["is_terminal"] = False
		
	def estimate_value(self, agent_id, states, concats=None, lstm_state=None):
		return self.get_model(agent_id).run_value(states=states, concats=concats, initial_lstm_state=lstm_state)
		
	def bootstrap(self, state, concat=None):
		agent_id = self.agent_id
		agent = self.get_model(agent_id)
		self.batch["bootstrap"] = {}
		self.batch["bootstrap"]["lstm_states"] = agent.lstm_state_out # do it BEFORE agent.run_policy_and_value
		value = agent.run_value(states=[state], concats=[concat])
		self.batch["bootstrap"]["agent_id"] = agent_id
		self.batch["bootstrap"]["state"] = state
		self.batch["bootstrap"]["concat"] = concat
		self.batch["bootstrap"]["value"] = value
		
	def act(self, policy_to_action_function, act_function, state, concat=None):
		agent_id = self.agent_id
		agent = self.get_model(agent_id)
		self.batch["states"][agent_id].append(state)
		self.batch["concats"][agent_id].append(concat)
		
		self.batch["lstm_states"][agent_id].append(agent.lstm_state_out) # do it BEFORE agent.run_policy_and_value
		policy, value = agent.run_policy_and_value(states=[state], concats=[concat])
		
		action = policy_to_action_function(policy)
		new_state, reward, terminal = act_function(action)
		if flags.clip_reward:
			reward = np.clip(reward, flags.min_reward, flags.max_reward)
		self.batch["total_reward"] += reward
		
		self.batch["rewards"][agent_id].append(reward)
		self.batch["values"][agent_id].append(value)
		self.batch["policies"][agent_id].append(policy)
		self.batch["actions"][agent_id].append(agent.get_action_vector(action))
		
		self.batch["agent-position_list"].append( (agent_id, len(self.batch["states"][agent_id])-1) ) # (agent_id, batch_position)
		self.batch["size"] += 1
		self.batch["is_terminal"] = terminal
		return new_state, policy, value, action, reward, terminal
		
	def get_frame(self, batch, index, keys=None):
		batch_size = batch["size"]
		if index >= batch_size or index < -batch_size:
			return None
		agent_id, batch_pos = batch["agent-position_list"][index]
		frame = {}
		if keys is None:
			frame["agent_id"] = agent_id
			for key in batch:
				frame[key] = batch[key][agent_id][batch_pos]
		else:
			for key in keys:
				if key == "agent_id":
					frame["agent_id"] = agent_id
				else:
					frame[key] = batch[key][agent_id][batch_pos]
		return frame
					
	def compute_cumulative_reward(self, batch):
		# prepare batch
		for i in range(self.model_size):
			batch["discounted_cumulative_rewards"][i]=deque()
			batch["generalized_advantage_estimators"][i]=deque()
		# bootstrap
		discounted_cumulative_reward = 0.0
		generalized_advantage_estimator = 0.0
		if "bootstrap" in batch:
			discounted_cumulative_reward = batch["bootstrap"]["value"]
		# compute cumulative reward and advantage
		last_value = discounted_cumulative_reward
		batch_size = batch["size"]
		for t in range(batch_size):
			frame = self.get_frame(batch=batch, index=-t-1, keys=["agent_id","values","rewards"])
			agent_id = frame["agent_id"]
			value = frame["values"]
			reward = frame["rewards"]
			discounted_cumulative_reward = reward + flags.gamma * discounted_cumulative_reward
			generalized_advantage_estimator = reward + flags.gamma * last_value - value + flags.gamma*flags.lambd*generalized_advantage_estimator
			last_value = value
			batch["discounted_cumulative_rewards"][agent_id].appendleft(discounted_cumulative_reward)
			batch["generalized_advantage_estimators"][agent_id].appendleft(generalized_advantage_estimator)
		return batch
		
	def train(self, batch):
		state=batch["states"]
		concat=batch["concats"]
		action=batch["actions"]
		value=batch["values"]
		policy=batch["policies"]
		reward=batch["rewards"]
		dcr=batch["discounted_cumulative_rewards"]
		gae=batch["generalized_advantage_estimators"]
		lstm_state=batch["lstm_states"]
		# assert self.global_network is not None, 'you are trying to train the global network'
		for i in range(self.model_size):
			batch_size = len(state[i])
			if batch_size > 0:
				self.get_model(i).train(
					states=state[i],
					concats=concat[i],
					actions=action[i],
					values=value[i],
					policies=policy[i],
					discounted_cumulative_rewards=dcr[i],
					generalized_advantage_estimators=gae[i],
					rewards=reward[i],
					lstm_states=lstm_state[i]
				)
				
	def replay_value(self, batch):
		for i in range(self.model_size):
			states = batch["states"][i]
			concats = batch["concats"][i]
			values = batch["values"][i]
			lstm_states = batch["lstm_states"][i]
			for j in range(len(states)):
				values[j] = self.estimate_value(
					agent_id=i, 
					states=[states[j]], 
					concats=[concats[j]], 
					lstm_state=lstm_states[j]
				)
		if "bootstrap" in batch:
			bootstrap = batch["bootstrap"]
			bootstrap["value"] = self.estimate_value(
				agent_id=bootstrap["agent_id"], 
				states=[bootstrap["state"]], 
				concats=[bootstrap["concat"]],
				lstm_state=bootstrap["lstm_states"]
			)
		return self.compute_cumulative_reward(batch)

	def process_batch(self):
		batch = self.compute_cumulative_reward(self.batch)
		self.train(batch)
		# experience replay
		if flags.replay_ratio > 0:
			if self.experience_buffer.has_atleast(flags.replay_start):
				n = np.random.poisson(flags.replay_ratio)
				for _ in range(n):
					(old_batch,_) = self.experience_buffer.get()
					self.train( self.replay_value(old_batch) if flags.replay_value else old_batch )
			batch_reward = batch["total_reward"]
			if batch_reward != 0 or not flags.save_only_batches_with_reward:
				self.experience_buffer.put(batch=batch, type=1 if batch_reward != 0 else 0)