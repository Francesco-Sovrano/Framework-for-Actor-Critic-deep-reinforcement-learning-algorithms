# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
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
		self._model_usage_list = collections.deque()
		self.build_agents(state_shape, action_size, concat_size)
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
			entropy_beta=flags.entropy_beta,
			clip=self.clip[0],
			concat_size=concat_size,
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

	def get_model( self, id ):
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
		for agent in self.model_list:
			agent.reset()
			
	def reset_batch(self):
		self.batch = {}
		self.batch["states"] = []
		self.batch["actions"] = []
		self.batch["concats"] = []
		self.batch["rewards"] = []
		self.batch["discounted_cumulative_rewards"] = []
		self.batch["generalized_advantage_estimators"] = []
		self.batch["values"] = []
		self.batch["policies"] = []
		self.batch["lstm_states"] = []
		
		for i in range(self.model_size):
			for key in self.batch:
				self.batch[key].append(collections.deque())
				
		self.agent_id_list = []
		self.reward_list = []
		self.value_list = []
		self.agent_id = 0
		self.step = 0 # start from 0
		self.batch_reward = 0
		
	def estimate_value(self, state, concat=None):
		agent = self.get_model(self.agent_id)
		return agent.run_value(state=[state], concat=[concat])
		
	def act(self, policy_to_action_function, act_function, state, concat=None):
		agent = self.get_model(self.agent_id)
		
		self.batch["lstm_states"][self.agent_id].append(agent.lstm_state_out) # do it before agent.run_policy_and_value
		self.batch["states"][self.agent_id].append(state)
		self.batch["concats"][self.agent_id].append(concat)
		
		policy, value = agent.run_policy_and_value(state=[state], concat=[concat])
		action = policy_to_action_function(policy)
		_, reward, terminal = act_function(action)
		if flags.clip_reward:
			reward = np.clip(reward, flags.min_reward, flags.max_reward)
		self.batch_reward += reward
		
		self.batch["rewards"][self.agent_id].append(reward)
		self.batch["values"][self.agent_id].append(value)
		self.batch["policies"][self.agent_id].append(policy)
		self.batch["actions"][self.agent_id].append(agent.get_action_vector(action))
		
		self.agent_id_list.append(self.agent_id)
		self.value_list.append(value) # we use it to calculate the GAE when out of this loop		
		self.reward_list.append(reward) # we use it to calculate the cumulative reward when out of this loop
		
		self.step += 1 # exec this command last
		return policy, value, action, reward, terminal
		
	def compute_cumulative_reward(self, state=None, concat=None):
		discounted_cumulative_reward = 0.0
		generalized_advantage_estimator = 0.0
		
		bootstrap = state is not None
		if bootstrap: # bootstrap the value from the last state
			agent = self.get_model(self.agent_id)
			discounted_cumulative_reward = agent.run_value(state=[state], concat=[concat])
			
		last_value = discounted_cumulative_reward
		batch_size = len(self.reward_list)
		for t in range(batch_size):
			index = batch_size-t-1
			agent_id = self.agent_id_list[index]
			value = self.value_list[index]
			reward = self.reward_list[index]
			discounted_cumulative_reward = reward + flags.gamma * discounted_cumulative_reward
			generalized_advantage_estimator = reward + flags.gamma * last_value - value + flags.gamma*flags.lambd*generalized_advantage_estimator
			last_value = value
			self.batch["discounted_cumulative_rewards"][agent_id].appendleft(discounted_cumulative_reward)
			self.batch["generalized_advantage_estimators"][agent_id].appendleft(generalized_advantage_estimator)
		
	def train(self, batch):
		state=batch["states"]
		action=batch["actions"]
		value=batch["values"]
		policy=batch["policies"]
		reward=batch["rewards"]
		dcr=batch["discounted_cumulative_rewards"]
		gae=batch["generalized_advantage_estimators"]
		lstm_state=batch["lstm_states"]
		concat=batch["concats"]
		# assert self.global_network is not None, 'you are trying to train the global network'
		for i in range(self.model_size):
			batch_size = len(state[i])
			if batch_size > 0:
				self.get_model(i).train(
					states=state[i],
					actions=action[i],
					values=value[i],
					policies=policy[i],
					discounted_cumulative_rewards=dcr[i],
					generalized_advantage_estimators=gae[i],
					rewards=reward[i],
					lstm_states=lstm_state[i],
					concat=concat[i]
				)
				
	def process_batch(self):
		self.train(batch=self.batch)
		# experience replay
		if flags.replay_ratio > 0:
			if self.experience_buffer.has_atleast(flags.replay_start):
				n = np.random.poisson(flags.replay_ratio)
				for _ in range(n):
					(batch,type) = self.experience_buffer.get()
					self.train(batch)
			if self.batch_reward != 0 or not flags.save_only_batches_with_reward:
				self.experience_buffer.put(batch=self.batch, type=1 if self.batch_reward != 0 else 0)