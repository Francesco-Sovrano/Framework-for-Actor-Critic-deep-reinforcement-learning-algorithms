# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from model.network.actor_critic_network import ActorCriticNetwork
from model.manager import BasicManager

import options
flags = options.get()

class ReinforcementLearningPartitioner(BasicManager):
	def set_model_size(self):
		self.model_size = flags.partition_count # manager output size
		if self.model_size < 2:
			self.model_size = 2
			
	def build_agents(self, state_shape, action_size, concat_size):
		agents_count = self.model_size-1
		# the manager
		self.manager = ActorCriticNetwork(session=self.session, id="{0}_{1}".format(self.id, 0), state_shape=state_shape, policy_size=agents_count, entropy_beta=flags.entropy_beta, clip=self.clip[0], device=self.device, concat_size=concat_size)
		self.model_list.append(self.manager)
		# the agents
		for i in range(agents_count):
			agent=ActorCriticNetwork(session=self.session, id="{0}_{1}".format(self.id, i+1), state_shape=state_shape, policy_size=action_size, entropy_beta=flags.entropy_beta*(i+1), clip=self.clip[i+1], device=self.device, concat_size=concat_size)
			self.model_list.append(agent)
			
	def initialize_gradient_optimizer(self):
		super().initialize_gradient_optimizer()
		initial_learning_rate = flags.alpha * flags.partitioner_learning_factor
		self.learning_rate[0] = eval('tf.train.'+flags.alpha_annealing_function)(learning_rate=initial_learning_rate, global_step=self.global_step[0], decay_steps=flags.alpha_decay_steps, decay_rate=flags.alpha_decay_rate) if flags.alpha_decay else initial_learning_rate
		self.gradient_optimizer[0] = eval('tf.train.'+flags.partitioner_optimizer+'Optimizer')(learning_rate=self.learning_rate[0], use_locking=True)
		
	def get_state_partition( self, state, concat=None ):
		policy, value = self.manager.run_policy_and_value( state, concat )
		id = np.random.choice(range(len(policy)), p=policy)
		self.add_to_statistics(id)
		id = id + 1 # the first agent is the manager
		return id, policy, value
		
	def query_partitioner(self, step):
		return step%flags.partitioner_granularity==0
		
	def reset_batch(self):
		super().reset_batch()
		self.manager_value_list = []
		
	def act(self, policy_to_action_function, act_function, state, concat=None):
		if self.query_partitioner(self.step):
			self.agent_id, manager_policy, manager_value = self.get_state_partition(state=[state], concat=[concat])
			self.batch["values"][0].append(manager_value)
			self.batch["policies"][0].append(manager_policy)
			self.batch["states"][0].append(state)
			self.batch["concat"][0].append(concat)
			self.batch["actions"][0].append(self.manager.get_action_vector( self.agent_id-1 ))
			self.manager_value_list.append(manager_value)			
		return super().act(policy_to_action_function, act_function, state, concat)
			
	def compute_cumulative_reward(self, state=None, concat=None):
		manager_discounted_cumulative_reward = 0.0
		manager_generalized_advantage_estimator = 0.0
		# Bootstrap partitioner
		bootstrap = state is not None
		if bootstrap: # bootstrap the value from the last state
			self.agent_id, _, manager_discounted_cumulative_reward = self.get_state_partition(state=[state], concat=[concat])
		# Compute agents' cumulative_reward	
		super().compute_cumulative_reward(state, concat)
		# Compute partitioner's cumulative_reward
		last_manager_value = manager_discounted_cumulative_reward
		batch_size = len(self.reward_list)
		query_reward = 0
		for t in range(batch_size):
			index = batch_size-t-1
			query_reward += self.reward_list[index]
			if self.query_partitioner(index): # ok because "step" starts from 0
				manager_value = self.manager_value_list.pop()
				manager_discounted_cumulative_reward = query_reward + flags.gamma * manager_discounted_cumulative_reward
				manager_generalized_advantage_estimator = query_reward + flags.gamma * last_manager_value - manager_value + flags.gamma*flags.lambd*manager_generalized_advantage_estimator
				last_manager_value = manager_value
				self.batch["discounted_cumulative_reward"][0].appendleft(manager_discounted_cumulative_reward)
				self.batch["generalized_advantage_estimator"][0].appendleft(manager_generalized_advantage_estimator)
				query_reward = 0