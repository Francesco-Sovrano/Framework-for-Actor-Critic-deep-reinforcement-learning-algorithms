# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import tensorflow as tf
import numpy as np
from agent.network import *
from agent.manager import BasicManager
from agent.batch import ExperienceBatch

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
		self.manager = eval('{}_Network'.format(flags.network))(
			session=self.session, 
			id='{0}_{1}'.format(self.id, 0), 
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
			agent = eval('{}_Network'.format(flags.network))(
				session=self.session, 
				id='{0}_{1}'.format(self.id, i+1), 
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
		action_batch, value_batch, entropy_batch, cross_entropy_batch, lstm_state = self.manager.run_action_and_value(states=[state], lstm_state=lstm_state)
		id = np.argwhere(action_batch[0]==1)[0][0]+1
		self.add_to_statistics(id)
		return id, action_batch[0], value_batch[0], entropy_batch[0], cross_entropy_batch[0], lstm_state
		
	def query_partitioner(self, step):
		return step%flags.partitioner_granularity==0
		
	def act(self, act_function, state, concat=None):
		if self.query_partitioner(self.batch.size):
			lstm_state = self.lstm_state
			self.agent_id, manager_action, manager_value, _, manager_cross_entropy, _ = self.get_state_partition(state=state, lstm_state=lstm_state)
			has_queried_partitioner = True
		else:
			has_queried_partitioner = False
			
		new_state, value, action, reward, terminal, cross_entropy, entropy = super().act(act_function, state, concat)
		if has_queried_partitioner:
			self.batch.add_agent_action(agent_id=0, state=state, concat=None, action=manager_action, cross_entropy=manager_cross_entropy, reward=reward, value=manager_value, lstm_state=lstm_state, memorize_step=False)
		return new_state, value, action, reward, terminal, cross_entropy, entropy
		
	def bootstrap(self, state, concat=None):
		id, _, value, _, _, _ = self.get_state_partition(state=state, lstm_state=self.lstm_state)
		if self.query_partitioner(self.batch.size):
			self.agent_id = id
		super().bootstrap(state, concat)
		self.batch.bootstrap['manager_value'] = value
			
	def replay_value(self, batch): # replay values, lstm states and partitions
		# init values
		new_batch = ExperienceBatch(self.model_size)
		# replay values
		lstm_state = batch.get_step_action('lstm_states', 0)
		for i in range(batch.size):
			state, concat, reward, action, cross_entropy = batch.get_step_action(['states','concats','rewards','actions','cross_entropys'], i)
			if self.query_partitioner(i):
				agent_id, manager_action, manager_value, _, manager_cross_entropy, _ = self.get_state_partition(state=state, lstm_state=lstm_state)
				new_batch.add_agent_action(agent_id=0, state=state, concat=None, action=manager_action, cross_entropy=manager_cross_entropy, reward=reward, value=manager_value, lstm_state=lstm_state, memorize_step=False)
			new_values, new_lstm_state = self.estimate_value(agent_id=agent_id, states=[state], concats=[concat], lstm_state=lstm_state)
			new_batch.add_agent_action(agent_id=agent_id, state=state, concat=concat, action=action, cross_entropy=cross_entropy, reward=reward, value=new_values[0], lstm_state=lstm_state, memorize_step=True)
			lstm_state = new_lstm_state
			
		if 'manager_value' in batch.bootstrap:
			new_batch.bootstrap = batch.bootstrap
			bootstrap = new_batch.bootstrap
			state = bootstrap['state']
			concat = bootstrap['concat']
			id, _, bootstrap['manager_value'], _, _, _ = self.get_state_partition(state=state, lstm_state=lstm_state)
			if self.query_partitioner(batch.size):
				agent_id = id
			values, _ = self.estimate_value(agent_id=agent_id, states=[state], concats=[concat], lstm_state=lstm_state)
			bootstrap['value'] = values[0]
		return self.compute_cumulative_reward(new_batch)
			
	def compute_cumulative_reward(self, batch):
		manager_discounted_cumulative_reward = 0.0
		manager_generalized_advantage_estimator = 0.0
		# Bootstrap partitioner
		if 'manager_value' in batch.bootstrap:
			manager_discounted_cumulative_reward = batch.bootstrap['manager_value']
		# Compute agents' cumulative_reward
		batch = super().compute_cumulative_reward(batch)
		# Compute partitioner's cumulative_reward
		last_manager_value = manager_discounted_cumulative_reward
		batch_size = batch.size
		query_reward = 0
		manager_value_list_idx = -1
		for t in range(batch_size):
			index = batch_size-t-1
			reward = batch.get_step_action('rewards', index)
			query_reward += reward
			if self.query_partitioner(index): # ok because 'step' starts from 0
				manager_value = batch.values[0][manager_value_list_idx]
				manager_value_list_idx -= 1
				
				manager_discounted_cumulative_reward = query_reward + flags.gamma * manager_discounted_cumulative_reward
				manager_generalized_advantage_estimator = query_reward + flags.gamma * last_manager_value - manager_value + flags.gamma*flags.lambd*manager_generalized_advantage_estimator
				last_manager_value = manager_value
				batch.discounted_cumulative_rewards[0].appendleft(manager_discounted_cumulative_reward)
				batch.generalized_advantage_estimators[0].appendleft(manager_generalized_advantage_estimator)
				query_reward = 0
		return batch