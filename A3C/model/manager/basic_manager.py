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
	def __init__(self, session, device, id, action_shape, state_shape, concat_size=0, global_network=None, training=True):
		self.training = training
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
		self.build_agents(state_shape=state_shape, action_shape=action_shape, concat_size=concat_size)
	# Build experience buffer
		if flags.replay_ratio > 0:
			self.experience_buffer = Buffer(size=flags.replay_buffer_size)
		if flags.predict_reward:
			self.reward_prediction_buffer = Buffer(size=flags.reward_prediction_buffer_size)
	# Bind optimizer to global
		if not self.is_global_network():
			self.bind_to_global(self.global_network)
			
	def is_global_network(self):
		return self.global_network is None
			
	def set_model_size(self):
		self.model_size = 1
			
	def build_agents(self, state_shape, action_shape, concat_size):
		agent=eval(flags.network + "_Network")(
			session=self.session,
			id="{0}_{1}".format(self.id, 0),
			device=self.device,
			state_shape=state_shape,
			action_shape=action_shape,
			concat_size=concat_size,
			entropy_beta=flags.entropy_beta,
			clip=self.clip[0],
			predict_reward=flags.predict_reward,
			training = self.training
		)
		self.model_list.append(agent)
			
	def sync(self):
		# assert not self.is_global_network(), 'you are trying to sync the global network with itself'
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
			self.sync_list.append(local_agent.bind_sync(global_agent)) # for syncing local network with global one

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
		self.lstm_state = None
			
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
		return self.get_model(agent_id).run_value(states=states, concats=concats, lstm_state=lstm_state)
		
	def act(self, act_function, state, concat=None):
		agent_id = self.agent_id
		agent = self.get_model(agent_id)
		self.batch["states"][agent_id].append(state)
		self.batch["concats"][agent_id].append(concat)
		
		self.batch["lstm_states"][agent_id].append(self.lstm_state) # do it before running policy and value
		policies, values, self.lstm_state = agent.run_policy_and_value(states=[state], concats=[concat], lstm_state=self.lstm_state)
		policy = policies[0]
		value = values[0]
		
		action, new_state, reward, terminal = act_function(policy)
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
		
	def get_frame(self, batch, index, keys=["states","concats","actions","rewards","values","policies","lstm_states","discounted_cumulative_rewards","generalized_advantage_estimators"]):
		batch_size = batch["size"]
		if index >= batch_size or index < -batch_size:
			return None
		agent_id, batch_pos = batch["agent-position_list"][index]
		frame = {}
		frame["agent_id"] = agent_id
		frame["batch_pos"] = batch_pos
		for key in keys:
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
			frame = self.get_frame(batch=batch, index=-t-1, keys=["values","rewards"])
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
		# assert self.global_network is not None, 'you are trying to train the global network'
		states = batch["states"]
		concats = batch["concats"]
		actions = batch["actions"]
		values = batch["values"]
		policies = batch["policies"]
		rewards = batch["rewards"]
		dcr = batch["discounted_cumulative_rewards"]
		gae = batch["generalized_advantage_estimators"]
		lstm_states = batch["lstm_states"]
		for i in range(self.model_size):
			batch_size = len(states[i])
			if batch_size > 0:
				model = self.get_model(i)
				# reward prediction
				if model.predict_reward:
					rp_batch = self.reward_prediction_buffer.get()
					rp_states, rp_target = rp_batch["states"], rp_batch["target"]
				else:
					rp_states = None
					rp_target = None
				# train
				model.train(
					states=states[i],
					concats=concats[i],
					actions=actions[i],
					values=values[i],
					policies=policies[i],
					discounted_cumulative_rewards=dcr[i],
					generalized_advantage_estimators=gae[i],
					rewards=rewards[i],
					lstm_state=lstm_states[i][0],
					reward_prediction_states=rp_states,
					reward_prediction_target=rp_target
				)
				
	def bootstrap(self, state, concat=None):
		values, _ = self.estimate_value(agent_id=self.agent_id, states=[state], concats=[concat], lstm_state=self.lstm_state)
		self.batch["bootstrap"] = {}
		self.batch["bootstrap"]["agent_id"] = self.agent_id
		self.batch["bootstrap"]["state"] = state
		self.batch["bootstrap"]["concat"] = concat
		self.batch["bootstrap"]["value"] = values[0]
				
	def replay_value(self, batch): # replay values, lstm states
		lstm_state = self.get_frame(batch=batch, index=0, keys=["lstm_states"])["lstm_states"]
		for i in range(batch["size"]):
			frame = self.get_frame(batch=batch, index=i, keys=["concats","states"])
			agent_id = frame["agent_id"]
			new_values, lstm_state = self.estimate_value(agent_id=agent_id, states=[frame["states"]], concats=[frame["concats"]], lstm_state=lstm_state)
			batch["values"][agent_id][frame["batch_pos"]] = new_values[0]
		if "bootstrap" in batch:
			bootstrap = batch["bootstrap"]
			values, _ = self.estimate_value(agent_id=bootstrap["agent_id"], states=[bootstrap["state"]], concats=[bootstrap["concat"]], lstm_state=lstm_state)
			bootstrap["value"] = values[0]
		return self.compute_cumulative_reward(batch)
		
	def should_save_batch(self, batch):
		# Prioritize smaller batches because they have terminated prematurely, thus they are probably more important and also faster to process
		batch_size = batch["size"]
		return np.random.randint(batch_size) == 0
		
	def add_to_reward_prediction_buffer(self, batch):
		batch_size = batch["size"]
		if batch_size <= 1:
			return
		rp_frame_length = min(3, batch_size-1)
		should_save_batch = self.should_save_batch(batch)
		for i in range(rp_frame_length, batch_size):
			reward = self.get_frame(batch=batch, index=i, keys=["rewards"])["rewards"]
			type_id = 1 if reward > 0 else 0
			if should_save_batch or not self.reward_prediction_buffer.id_is_full(type_id):
				states = [self.get_frame(batch=batch, index=j, keys=["states"])["states"] for j in range(i-rp_frame_length, i)]
				target = [0.0, 0.0, 0.0]
				if reward == 0:
					target[0] = 1.0 # zero
				elif reward > 0:
					target[1] = 1.0 # positive
				else:
					target[2] = 1.0 # negative
				# if len(states)==0:
					# print(i, " ", rp_frame_length)
				self.reward_prediction_buffer.put(batch={"states":states, "target":[target]}, type_id=type_id)
			
	def add_to_replay_buffer(self, batch):
		batch_size = batch["size"]
		if batch_size <= 1:
			return
		batch_reward = batch["total_reward"]
		if batch_reward == 0 and flags.save_only_batches_with_reward:
			return
		type_id = 1 if batch_reward > 0 else 0
		if not self.experience_buffer.id_is_full(type_id) or self.should_save_batch(batch):
			# batch["lstm_states"] = None # remove lstm state from batch
			self.experience_buffer.put(batch=batch, type_id=type_id)
		
	def process_batch(self, global_step):
		batch = self.compute_cumulative_reward(self.batch)
		# reward prediction
		if flags.predict_reward:
			self.add_to_reward_prediction_buffer(batch) # do it before training, this way there will be at least one batch in the reward_prediction_buffer
		# train
		self.train(batch)
		# experience replay
		if flags.replay_ratio > 0 and global_step > flags.replay_step:
			if self.experience_buffer.has_atleast(flags.replay_start):
				n = np.random.poisson(flags.replay_ratio)
				for _ in range(n):
					old_batch = self.experience_buffer.get()
					self.train( self.replay_value(old_batch) if flags.replay_value else old_batch )
			self.add_to_replay_buffer(batch)