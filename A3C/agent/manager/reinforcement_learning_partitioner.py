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

	def get_agents_count(self):
		return self.model_size-1
			
	def build_agents(self, state_shape, action_shape, concat_size):
		agents_count = self.get_agents_count()
		# the manager
		self.manager = eval('{}_Network'.format(flags.network))(
			id='{0}_{1}'.format(self.id, 0), 
			device=self.device, 
			session=self.session, 
			state_shape=state_shape, 
			action_shape=(1,agents_count), 
			concat_size=agents_count+1 if flags.use_concatenation else 0, 
			beta=flags.partitioner_beta, 
			gamma=flags.partitioner_gamma, 
			clip=self.clip[0], 
			predict_reward=flags.predict_reward, 
			training = self.training
		)
		self.model_list.append(self.manager)
		# the agents
		for i in range(agents_count):
			agent = eval('{}_Network'.format(flags.network))(
				id='{0}_{1}'.format(self.id, i+1), 
				device=self.device, 
				session=self.session, 
				state_shape=state_shape, 
				action_shape=action_shape, 
				concat_size=concat_size, 
				beta=flags.beta + i*flags.beta_translation_per_agent, 
				gamma=flags.gamma + i*flags.gamma_translation_per_agent, 
				clip=self.clip[i+1], 
				predict_reward=flags.predict_reward, 
				training = self.training, 
				parent = self.manager, 
				sibling = self.model_list[1] if i > 0 else None # the first agent (non manager)
			)
			self.model_list.append(agent)
			
	def initialize_gradient_optimizer(self):
		super().initialize_gradient_optimizer()
		initial_learning_rate = flags.partitioner_alpha
		self.learning_rate[0] = eval('tf.train.'+flags.alpha_annealing_function)(learning_rate=initial_learning_rate, global_step=self.global_step[0], decay_steps=flags.alpha_decay_steps, decay_rate=flags.alpha_decay_rate) if flags.alpha_decay else initial_learning_rate
		# self.learning_rate[0] *= flags.partitioner_learning_factor
		# self.gradient_optimizer[0] = eval('tf.train.'+flags.partitioner_optimizer+'Optimizer')(learning_rate=self.learning_rate[0], use_locking=True)
		
	def get_state_partition(self, state, concat=None):
		action_batch, value_batch, policy_batch = self.manager.predict_action(states=[state], concats=[concat])
		id = np.argwhere(action_batch[0]==1)[0][0]+1
		self.add_to_statistics(id)
		return id, action_batch[0], value_batch[0], policy_batch[0]
		
	def query_partitioner(self, step):
		return step%flags.partitioner_granularity==0
		
	def get_manager_concatenation(self):
		return np.concatenate((self.last_manager_action, [self.last_manager_reward]), -1)
		
	def reset(self):
		super().reset()
		self.last_manager_action = [0]*self.get_agents_count()
		self.last_manager_reward = 0
		
	def act(self, act_function, state, concat=None):
		if self.query_partitioner(self.batch.size):
			manager_concat = self.get_manager_concatenation()
			self.agent_id, manager_action, manager_value, manager_policy = self.get_state_partition(state=state, concat=manager_concat)
			self.last_manager_action = manager_action
			# N.B.: the query reward is unknown since bootstrap or a new query starts
			self.batch.add_agent_action(agent_id=0, state=state, concat=manager_concat, action=manager_action, policy=manager_policy, reward=0, value=manager_value, memorize_step=False)
			
		new_state, value, action, reward, terminal, policy = super().act(act_function, state, concat)
		# keep query reward updated
		self.last_manager_reward += reward
		self.batch.rewards[0][-1] = self.last_manager_reward
		return new_state, value, action, reward, terminal, policy
		
	def bootstrap(self, state, concat=None):
		manager_concat = self.get_manager_concatenation()
		new_agent_id, _, manager_value, _ = self.get_state_partition(state=state, concat=manager_concat)
		self.batch.bootstrap['manager_concat'] = manager_concat
		self.batch.bootstrap['manager_value'] = manager_value
		if self.query_partitioner(self.batch.size):
			self.agent_id = new_agent_id
		super().bootstrap(state, concat)
		
	def replay_value(self, batch): # replay values
		for i in range(len(batch.states[0])):
			state = batch.states[0][i]
			concat = batch.concats[0][i]
			batch.values[0][i] = self.estimate_value(agent_id=0, states=[state], concats=[concat])[0]
		if 'manager_value' in batch.bootstrap:
			bootstrap = batch.bootstrap
			bootstrap['manager_value'] = self.estimate_value(agent_id=0, states=[bootstrap['state']], concats=[bootstrap['manager_concat']])[0]
		return super().replay_value(batch)
		
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
		batch_length = len(batch.values[0])
		for i in range(batch_length-1,-1,-1):
			query_reward = batch.rewards[0][i]
			manager_value = batch.values[0][i]
			manager_discounted_cumulative_reward = query_reward + self.manager.gamma*manager_discounted_cumulative_reward
			manager_generalized_advantage_estimator = query_reward + self.manager.gamma*last_manager_value - manager_value + self.manager.gamma*flags.lambd*manager_generalized_advantage_estimator
			last_manager_value = manager_value
			batch.discounted_cumulative_rewards[0].appendleft(manager_discounted_cumulative_reward)
			batch.generalized_advantage_estimators[0].appendleft(manager_generalized_advantage_estimator)
		return batch