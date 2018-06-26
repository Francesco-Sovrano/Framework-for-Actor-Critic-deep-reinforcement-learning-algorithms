# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
import numpy as np
from model.a3c_model import A3CModel
from model.experience_buffer import Buffer

import options
flags = options.get()

class ModelManager(object):
	def __init__(self, id, environment, device, global_network=None):
		self.id = id
		self.device = device
		self.environment = environment
		self.global_network = global_network
		# get information to build agents
		agents_count = flags.partition_count # manager output size
		action_size = self.environment.get_action_size() # agent output size
		state_shape = self.environment.get_state_shape() # input size
		if agents_count > 1:
			self.model_size = agents_count+1 # need for 1 extra agent as manager
			self.has_manager = True
		else:
			self.model_size = 1
			self.has_manager = False
			
	# Gradient optimizer and clip range
		if not(self.global_network is None):
			self.clip = self.global_network.clip
		else:
			self.initialize_gradient_optimizer()
			
	# Build agents
		self.model_list = []
		self._model_usage_matrix = np.zeros(agents_count)
		self._model_usage_list = collections.deque()		
		if self.has_manager:
			# the manager
			self.manager = A3CModel(id=str(self.id)+"_0", state_shape=state_shape, policy_size=agents_count, entropy_beta=flags.entropy_beta, clip=self.clip[0], device=self.device, concat_size=action_size+1)
			self.model_list.append(self.manager)
			# the agents
			for i in range(agents_count):
				agent=A3CModel(id=str(self.id)+"_"+str(i+1), state_shape=state_shape, policy_size=action_size, entropy_beta=flags.entropy_beta*(i+1), clip=self.clip[i+1], device=self.device, concat_size=action_size+1)
				self.model_list.append(agent)
		else:
			agent=A3CModel(id=str(self.id)+"_0", state_shape=state_shape, policy_size=action_size, entropy_beta=flags.entropy_beta, clip=self.clip[0], device=self.device, concat_size=action_size+1)
			self.model_list.append(agent)
			
	# Build experience buffer
		if flags.replay_ratio > 0:
			self.experience_buffer = Buffer(size=flags.replay_size)
		
	# Bind optimizer
		if not(self.global_network is None):
			self.apply_gradients, self.sync = self.bind_to_global(self.global_network)
			
	def sync_with_global(self, session):
		# assert self.global_network is not None, 'you are trying to sync the global network with itself'
		for i in range(self.model_size):
			session.run(self.sync[i])
			
	def initialize_gradient_optimizer(self):
		self.global_step = []
		self.learning_rate = []
		self.clip = []
		self.gradient_optimizer = []
		for i in range(self.model_size):
		# global step
			self.global_step.append( tf.Variable(0, trainable=False) )
		# learning rate
			initial_learning_rate = flags.alpha
			if i == 0 and self.model_size > 1:
				initial_learning_rate *= flags.partitioner_learning_factor
			self.learning_rate.append( eval('tf.train.'+flags.alpha_annealing_function)(learning_rate=initial_learning_rate, global_step=self.global_step[i], decay_steps=flags.alpha_decay_steps, decay_rate=flags.alpha_decay_rate) if flags.alpha_decay else initial_learning_rate )
		# clip
			self.clip.append( eval('tf.train.'+flags.clip_annealing_function)(learning_rate=flags.clip, global_step=self.global_step[i], decay_steps=flags.clip_decay_steps, decay_rate=flags.clip_decay_rate) if flags.clip_decay else flags.clip )
		# gradient optimizer
			if i == 0 and self.model_size > 1:
				self.gradient_optimizer.append( eval('tf.train.'+flags.partitioner_optimizer+'Optimizer')(learning_rate=self.learning_rate[0], use_locking=True) )
			else:	
				self.gradient_optimizer.append( eval('tf.train.'+flags.optimizer+'Optimizer')(learning_rate=self.learning_rate[i], use_locking=True) )
			
	def bind_to_global(self, global_network):
		apply_gradients = []
		sync = []
		for i in range(self.model_size):
			local_agent = self.get_model(i)
			global_agent = global_network.get_model(i)
			local_agent.prepare_loss()
			apply_gradients.append( local_agent.minimize_local(optimizer=global_network.gradient_optimizer[i], global_step=global_network.global_step[i], global_var_list=global_agent.get_vars()) )
			sync.append( local_agent.sync_from(global_agent) ) # for synching local network with global one
		return apply_gradients, sync

	def get_model( self, id ):
		return self.model_list[id]
		
	def get_statistics(self):
		stats = {}
		if self.has_manager:
			total_usage = 0
			usage_matrix = np.zeros(self.model_size-1, dtype=np.uint16)
			for u in self._model_usage_list:
				usage_matrix[u] += 1
				total_usage += 1
			for i in range(self.model_size-1):
				stats["model_{0}".format(i)] = usage_matrix[i]/total_usage if total_usage != 0 else 0
		return stats
		
	def get_agentID_by_state( self, sess, state, concat=None ):
		if self.has_manager:
			policy, value = self.manager.run_policy_and_value( sess, state, concat )
			id = np.random.choice(range(len(policy)), p=policy)
			self._model_usage_matrix[id] += 1
			self._model_usage_list.append(id)
			if len(self._model_usage_list) > flags.match_count_for_evaluation:
				self._model_usage_list.popleft()

			id = id + 1 # the first agent is the manager
			return id, policy, value
		else:
			return 0, None, None
		
	def get_vars(self):
		vars = []
		for agent in self.model_list:
			vars = set().union(agent.get_vars(),vars)
		return list(vars)
		
	def reset(self):
		for agent in self.model_list:
			agent.reset_LSTM_state() # reset LSTM state
			
	def reset_batch(self):
		self.batch = {}
		self.batch["states"] = []
		self.batch["actions"] = []
		self.batch["concat"] = []
		self.batch["discounted_cumulative_reward"] = []
		self.batch["generalized_advantage_estimator"] = []
		self.batch["values"] = []
		self.batch["policies"] = []
		self.batch["start_lstm_state"] = []
		for i in range(self.model_size):
			for key in self.batch:
				self.batch[key].append(collections.deque())
			self.batch["start_lstm_state"][i] = self.get_model(i).lstm_state_out				
		self.agent_id_list = collections.deque()
		self.agent_reward_list = collections.deque()
		self.agent_value_list = collections.deque()
		self.agent_id = 0
		self.step = 0 # start from 0
		
	def query_partitioner(self, step):
		return step%flags.partitioner_granularity==0
		
	def act(self, session):
		state = self.environment.last_state
		concat = self.environment.get_last_action_reward()
		
		if self.has_manager:
			query_partitioner = self.query_partitioner(self.step)
			if query_partitioner:
				self.agent_id, manager_policy, manager_value = self.get_agentID_by_state(sess=session, state=[state], concat=[concat])
		agent = self.get_model(self.agent_id)
		agent_policy, agent_value = agent.run_policy_and_value(sess=session, state=[state], concat=[concat])
		
		action = self.environment.choose_action(agent_policy)
		new_state, reward, terminal = self.environment.process(action)
		if flags.clip_reward:
			reward = np.clip(reward, flags.min_reward, flags.max_reward)

		self.batch["states"][self.agent_id].append(state)
		self.batch["concat"][self.agent_id].append(concat)
		self.batch["values"][self.agent_id].append(agent_value)
		self.batch["policies"][self.agent_id].append(agent_policy)
		self.batch["actions"][self.agent_id].append(agent.get_action_vector(action))
		self.agent_reward_list.appendleft(reward) # we use it to calculate the cumulative reward when out of this loop
		self.agent_value_list.appendleft(agent_value) # we use it to calculate the GAE when out of this loop		
		# Populate manager self.batch
		if self.has_manager:
			self.agent_id_list.appendleft(self.agent_id)
			if query_partitioner:
				self.batch["states"][0].append(state)
				self.batch["actions"][0].append(self.manager.get_action_vector( self.agent_id-1 ))
				self.batch["concat"][0].append(concat)
				self.batch["values"][0].append(manager_value)
				self.batch["policies"][0].append(manager_policy)
				
		self.step += 1 # exec this command last
		return reward, terminal
			
	def save_batch(self, session, bootstrap):
		discounted_cumulative_reward = 0.0
		generalized_advantage_estimator = 0.0
		
		if bootstrap: # bootstrap the value from the last state
			state = self.environment.last_state
			last_action_reward = self.environment.get_last_action_reward()
			if self.has_manager and self.query_partitioner(self.step):
				self.agent_id, _, _ = self.get_agentID_by_state(sess=session, state=[state], concat=[last_action_reward])
			
			agent = self.get_model(self.agent_id)
			discounted_cumulative_reward = agent.run_value(sess=session, state=[state], concat=[last_action_reward])
			
		last_agent_value = discounted_cumulative_reward
		batch_reward = 0
		batch_size = len(self.agent_reward_list)
		for t in range(batch_size):
			agent_reward = self.agent_reward_list[t]
			agent_value = self.agent_value_list[t]
			
			batch_reward += agent_reward
			discounted_cumulative_reward = agent_reward + flags.gamma * discounted_cumulative_reward
			generalized_advantage_estimator = agent_reward + flags.gamma * last_agent_value - agent_value + flags.gamma*flags.lambd*generalized_advantage_estimator
			last_agent_value = agent_value

			if not self.has_manager or self.query_partitioner(batch_size-t-1): # ok because "step" starts from 0
				self.batch["discounted_cumulative_reward"][0].appendleft(discounted_cumulative_reward)
				self.batch["generalized_advantage_estimator"][0].appendleft(generalized_advantage_estimator)
			if self.has_manager:
				agent_id = self.agent_id_list[t]
				self.batch["discounted_cumulative_reward"][agent_id].appendleft(discounted_cumulative_reward)
				self.batch["generalized_advantage_estimator"][agent_id].appendleft(generalized_advantage_estimator)
		self.train(session=session, batch=self.batch)
		# experience replay
		if flags.replay_ratio > 0:
			if self.experience_buffer.has_atleast(flags.replay_start):
				n = np.random.poisson(flags.replay_ratio)
				for _ in range(n):
					self.train(session=session, batch=self.experience_buffer.get())
			if batch_reward != 0 or not flags.save_only_batches_with_reward:
				self.experience_buffer.put(batch=self.batch)
		
	def train(self, session, batch):
		state=batch["states"]
		action=batch["actions"]
		value=batch["values"]
		policy=batch["policies"]
		reward=batch["discounted_cumulative_reward"]
		gae=batch["generalized_advantage_estimator"]
		lstm_state=batch["start_lstm_state"]
		concat=batch["concat"]
		# assert self.global_network is not None, 'you are trying to train the global network'
		for i in range(self.model_size):
			if len(state[i]) > 0:
				self.get_model(i).train(session, self.apply_gradients[i], state[i], action[i], value[i], policy[i], reward[i], gae[i], lstm_state[i], concat[i])