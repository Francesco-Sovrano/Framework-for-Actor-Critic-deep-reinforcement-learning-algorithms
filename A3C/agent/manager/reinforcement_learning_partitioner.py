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
		self.agents_set = set(range(1,self.model_size))

	def get_agents_count(self):
		return self.model_size-1
			
	def build_agents(self, state_shape, action_shape, concat_size):
		agents_count = self.get_agents_count()
		# the manager
		manager = eval('{}_Network'.format(flags.network))(
			id='{0}_{1}'.format(self.id, 0), 
			device=self.device, 
			session=self.session, 
			state_shape=state_shape, 
			action_shape=(1,agents_count), 
			# concat_size=agents_count+2 if flags.use_concatenation else 0, 
			beta=flags.partitioner_beta, 
			clip=self.clip[0], 
			predict_reward=flags.predict_reward, 
			training = self.training
		)
		self.model_list.append(manager)
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
				clip=self.clip[i+1], 
				predict_reward=flags.predict_reward, 
				training = self.training, 
				parent = manager, 
				sibling = self.model_list[1] if i > 0 else None # the first agent (non manager)
			)
			self.model_list.append(agent)
			
	def initialize_gradient_optimizer(self):
		super().initialize_gradient_optimizer()
		initial_learning_rate = flags.partitioner_alpha
		self.learning_rate[0] = eval('tf.train.'+flags.alpha_annealing_function)(learning_rate=initial_learning_rate, global_step=self.global_step[0], decay_steps=flags.alpha_decay_steps, decay_rate=flags.alpha_decay_rate) if flags.alpha_decay else initial_learning_rate
		# self.learning_rate[0] *= flags.partitioner_learning_factor
		self.gradient_optimizer[0] = eval('tf.train.'+flags.partitioner_optimizer+'Optimizer')(learning_rate=self.learning_rate[0], use_locking=True)
		
	def get_state_partition(self, state, concat=None, internal_state=None):
		action_batch, value_batch, policy_batch, new_internal_state = self.get_model(0).predict_action(states=[state], concats=[concat], internal_state=internal_state)
		id = np.argwhere(action_batch[0]==1)[0][0]+1
		self.add_to_statistics(id)
		return id, action_batch[0], value_batch[0], policy_batch[0], new_internal_state
		
	def query_partitioner(self, step):
		return step%flags.partitioner_granularity==0
		
	def get_manager_concatenation(self):
		return np.concatenate((self.last_manager_action,self.last_manager_reward), -1)
		
	def reset(self):
		super().reset()
		self.last_manager_action = [0]*self.get_agents_count()
		# self.last_manager_value = 0
		self.last_manager_reward = np.zeros(2)
		self.manager_internal_state = None
		
	def initialize_new_batch(self):
		query_partitioner = self.query_partitioner(self.step)
		if not query_partitioner:
			last_action = self.batch.get_action(['states','concats','actions','policies','rewards','values','internal_states'], 0, -1)
		super().initialize_new_batch()
		if not query_partitioner:
			state, concat, action, policy, reward, value, internal_state = last_action
			self.batch.add_action(agent_id=0, state=state, concat=concat, action=action, policy=policy, reward=reward, value=value, internal_state=internal_state)
		
	def act(self, act_function, state, concat=None):
		if self.query_partitioner(self.step):
			internal_state = self.manager_internal_state
			manager_concat = self.get_manager_concatenation()
			self.agent_id, manager_action, manager_value, manager_policy, self.manager_internal_state = self.get_state_partition(state=state, concat=manager_concat, internal_state=internal_state)
			self.last_manager_action = manager_action
			# self.last_manager_value = manager_value
			self.last_manager_reward = np.zeros(2) # [extrinsic, intrinsic] # N.B.: the query reward is unknown since bootstrap or a new query starts
			if self.training:
				self.batch.add_action(agent_id=0, state=state, concat=manager_concat, action=manager_action, policy=manager_policy, reward=self.last_manager_reward, value=manager_value, internal_state=internal_state)
			
		new_state, value, action, total_reward, terminal, policy = super().act(act_function, state, concat)
		# keep query reward updated
		self.last_manager_reward += total_reward
		if self.training:
			self.batch.set_action({'rewards':self.last_manager_reward}, 0, -1)
		return new_state, value, action, total_reward, terminal, policy
		
	def bootstrap(self, state, concat=None):
		manager_concat = self.get_manager_concatenation()
		new_agent_id, _, manager_value, _, _ = self.get_state_partition(state=state, concat=manager_concat, internal_state=self.manager_internal_state)
		self.batch.bootstrap['manager_internal_state'] = self.manager_internal_state
		self.batch.bootstrap['manager_concat'] = manager_concat
		self.batch.bootstrap['manager_value'] = manager_value
		if self.query_partitioner(self.step):
			self.agent_id = new_agent_id
		super().bootstrap(state, concat)
		
	def replay_value(self, batch): # replay values
		if 'manager_value' in batch.bootstrap: # do it before calling super().replay_value(batch)
			bootstrap = batch.bootstrap
			value_batch, _ = self.estimate_value(agent_id=0, states=[bootstrap['state']], concats=[bootstrap['manager_concat']], internal_state=bootstrap['manager_internal_state'])
			bootstrap['manager_value'] = value_batch[0]
		return super().replay_value(batch)
		
	def compute_discounted_cumulative_reward(self, batch):
		batch.compute_discounted_cumulative_reward(agents=self.agents_set, last_value=batch.bootstrap['value'] if 'value' in batch.bootstrap else 0., gamma=flags.gamma, lambd=flags.lambd)
		batch.compute_discounted_cumulative_reward(agents=[0], last_value=batch.bootstrap['manager_value'] if 'manager_value' in batch.bootstrap else 0., gamma=flags.partitioner_gamma, lambd=flags.lambd)
		return batch