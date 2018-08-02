# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import tensorflow as tf
import numpy as np
from model.network import *
from model.manager import BasicManager
from model.batch import ExperienceBatch

import options
flags = options.get()

class ReinforcementLearningPartitioner(BasicManager):
	def set_model_size(self):
		self.model_size = flags.partition_count+1 # manager output size
		if self.model_size < 2:
			self.model_size = 2
			
	def build_agents(self, state_shape, action_shape, concat_size):
		agents_count = self.model_size-1
		# the manager
		self.manager = eval(flags.network + "_Network")(
			session=self.session, 
			id="{0}_{1}".format(self.id, 0), 
			state_shape=state_shape, 
			action_shape=(agents_count,1), 
			entropy_beta=flags.entropy_beta, 
			clip=self.clip[0], 
			device=self.device, 
			predict_reward=flags.predict_reward,
			training = self.training
		)
		self.model_list.append(self.manager)
		# the agents
		for i in range(agents_count):
			agent = eval(flags.network + "_Network")(
				session=self.session, 
				id="{0}_{1}".format(self.id, i+1), 
				state_shape=state_shape, 
				action_shape=action_shape, 
				concat_size=concat_size,
				entropy_beta=flags.entropy_beta*(i+1), 
				clip=self.clip[i+1], 
				device=self.device, 
				predict_reward=flags.predict_reward,
				training = self.training
			)
			self.model_list.append(agent)
			
	def initialize_gradient_optimizer(self):
		super().initialize_gradient_optimizer()
		initial_learning_rate = flags.alpha * flags.partitioner_learning_factor
		self.learning_rate[0] = eval('tf.train.'+flags.alpha_annealing_function)(learning_rate=initial_learning_rate, global_step=self.global_step[0], decay_steps=flags.alpha_decay_steps, decay_rate=flags.alpha_decay_rate) if flags.alpha_decay else initial_learning_rate
		self.gradient_optimizer[0] = eval('tf.train.'+flags.partitioner_optimizer+'Optimizer')(learning_rate=self.learning_rate[0], use_locking=True)
		
	def get_state_partition(self, state, lstm_state=None):
		policies, values, lstm_state = self.manager.run_policy_and_value(states=[state], lstm_state=lstm_state)
		policy = policies[0]
		value = values[0]
		id = np.random.choice(range(len(policy)), p=policy) + 1 # the first agent is the manager
		self.add_to_statistics(id)
		return id, policy, value, lstm_state
		
	def query_partitioner(self, step):
		return step%flags.partitioner_granularity==0
		
	def initialize_new_batch(self):
		super().initialize_new_batch()
		self.batch.manager_value_list = []
		
	def act(self, act_function, state, concat=None):
		if self.query_partitioner(self.batch.size):
			self.batch.lstm_states[0].append(self.lstm_state)
			self.agent_id, manager_policy, manager_value, _ = self.get_state_partition(state=state, lstm_state=self.lstm_state)
			
			self.batch.values[0].append(manager_value)
			self.batch.policies[0].append(manager_policy)
			self.batch.states[0].append(state)
			self.batch.concats[0].append(None)
			self.batch.actions[0].append(self.manager.get_action_vector(self.agent_id-1))
			self.batch.manager_value_list.append(manager_value)
			has_queried_partitioner = True
		else:
			has_queried_partitioner = False
			
		new_state, policy, value, action, reward, terminal = super().act(act_function, state, concat)
		if has_queried_partitioner:
			self.batch.rewards[0].append(reward)
		return new_state, policy, value, action, reward, terminal
		
	def bootstrap(self, state, concat=None):
		id, _, value, _ = self.get_state_partition(state=state, lstm_state=self.lstm_state)
		if self.query_partitioner(self.batch.size):
			self.agent_id = id
		super().bootstrap(state, concat)
		self.batch.bootstrap["manager_value"] = value
			
	def replay_value(self, batch): # replay values, lstm states and partitions
		# init values
		new_batch = ExperienceBatch(self.model_size)
		new_batch.states[0] = batch.states[0]
		new_batch.concats[0] = batch.concats[0]
		new_batch.rewards[0] = batch.rewards[0]
		new_batch.total_reward = batch.total_reward
		new_batch.size = batch.size
		new_batch.is_terminal = batch.is_terminal
		new_batch.manager_value_list = []
		# replay values
		id, pos = batch.get_agent_and_pos(index=0)
		lstm_state = batch.lstm_states[id][pos]
		for i in range(batch.size):
			id, pos = batch.get_agent_and_pos(index=i)
			state = batch.states[id][pos]
			concat = batch.concats[id][pos]
			if self.query_partitioner(i):
				new_batch.lstm_states[0].append(lstm_state)
				agent_id, manager_policy, manager_value, _ = self.get_state_partition(state=state, lstm_state=lstm_state)
				new_batch.values[0].append(manager_value)
				new_batch.policies[0].append(manager_policy)
				new_batch.actions[0].append(self.manager.get_action_vector(agent_id-1))
				new_batch.manager_value_list.append(manager_value)
			new_batch.states[agent_id].append(state)
			new_batch.concats[agent_id].append(concat)
			new_batch.rewards[agent_id].append(batch.rewards[id][pos])
			new_batch.policies[agent_id].append(batch.policies[id][pos])
			new_batch.actions[agent_id].append(batch.actions[id][pos])
			new_batch.lstm_states[agent_id].append(lstm_state)
			new_values, lstm_state = self.estimate_value(agent_id=agent_id, states=[state], concats=[concat], lstm_state=lstm_state)
			new_batch.values[agent_id].append(new_values[0])
			new_batch.agent_position_list.append( (agent_id, len(new_batch.states[agent_id])-1) ) # (agent_id, batch_position)
			
		if len(batch.bootstrap) > 0:
			new_batch.bootstrap = batch.bootstrap
			bootstrap = new_batch.bootstrap
			state = bootstrap["state"]
			concat = bootstrap["concat"]
			id, _, bootstrap["manager_value"], _ = self.get_state_partition(state=state, lstm_state=lstm_state)
			if self.query_partitioner(batch.size):
				agent_id = id
			values, _ = self.estimate_value(agent_id=agent_id, states=[state], concats=[concat], lstm_state=lstm_state)
			bootstrap["value"] = values[0]
		return self.compute_cumulative_reward(new_batch)
			
	def compute_cumulative_reward(self, batch):
		manager_discounted_cumulative_reward = 0.0
		manager_generalized_advantage_estimator = 0.0
		# Bootstrap partitioner
		if "manager_value" in batch.bootstrap:
			manager_discounted_cumulative_reward = batch.bootstrap["manager_value"]
		# Compute agents' cumulative_reward
		batch = super().compute_cumulative_reward(batch)
		# Compute partitioner's cumulative_reward
		last_manager_value = manager_discounted_cumulative_reward
		batch_size = batch.size
		query_reward = 0
		manager_value_list_idx = -1
		for t in range(batch_size):
			index = batch_size-t-1
			id, pos = batch.get_agent_and_pos(index)
			reward = batch.rewards[id][pos]
			query_reward += reward
			if self.query_partitioner(index): # ok because "step" starts from 0
				manager_value = batch.manager_value_list[manager_value_list_idx]
				manager_value_list_idx -= 1
				
				manager_discounted_cumulative_reward = query_reward + flags.gamma * manager_discounted_cumulative_reward
				manager_generalized_advantage_estimator = query_reward + flags.gamma * last_manager_value - manager_value + flags.gamma*flags.lambd*manager_generalized_advantage_estimator
				last_manager_value = manager_value
				batch.discounted_cumulative_rewards[0].appendleft(manager_discounted_cumulative_reward)
				batch.generalized_advantage_estimators[0].appendleft(manager_generalized_advantage_estimator)
				query_reward = 0
		return batch