# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import tensorflow as tf
import numpy as np
from agent.network import *
from utils.buffer import Buffer, PrioritizedBuffer
# from utils.schedules import LinearSchedule
from agent.batch import ExperienceBatch
from sklearn import random_projection

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
		self.build_agents(state_shape=state_shape, action_shape=action_shape, concat_size=concat_size)
	# Build experience buffer
		if flags.replay_ratio > 0:
			if flags.prioritized_replay:
				self.experience_buffer = PrioritizedBuffer(size=flags.replay_buffer_size)
				# self.beta_schedule = LinearSchedule(flags.max_time_step, initial_p=0.4, final_p=1.0)
			else:
				self.experience_buffer = Buffer(size=flags.replay_buffer_size)
		if flags.predict_reward:
			self.reward_prediction_buffer = Buffer(size=flags.reward_prediction_buffer_size)
	# Bind optimizer to global
		if not self.is_global_network():
			self.bind_to_global(self.global_network)
	# Statistics
		self._model_usage_list = deque()
		if flags.print_loss:
			self._loss_list = [{} for _ in range(self.model_size)]
	# Count based exploration	
		if flags.use_count_based_exploration_reward:
			self.projection = None
			self.projection_train_set = []
			
	def is_global_network(self):
		return self.global_network is None
			
	def set_model_size(self):
		self.model_size = 1
		self.agents_set = set([1])
			
	def build_agents(self, state_shape, action_shape, concat_size):
		agent=eval('{}_Network'.format(flags.network))(
			session=self.session, 
			id='{0}_{1}'.format(self.id, 0), 
			device=self.device, 
			state_shape=state_shape, 
			action_shape=action_shape, 
			concat_size=concat_size, 
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
			local_agent.minimize_local_loss(optimizer=global_network.gradient_optimizer[i], global_step=global_network.global_step[i], global_var_list=global_agent.get_shared_keys())
			self.sync_list.append(local_agent.bind_sync(global_agent)) # for syncing local network with global one

	def get_model(self, id):
		return self.model_list[id]
		
	def get_statistics(self):
		stats = {}
		# build loss statistics
		if flags.print_loss:
			for i in range(self.model_size):
				for key, value in self._loss_list[i].items():
					stats['loss_{}{}_avg'.format(key,i)] = np.average(value)
		# build models usage statistics
		if self.model_size > 1:
			total_usage = 0
			usage_matrix = {}
			for u in self._model_usage_list:
				if not (u in usage_matrix):
					usage_matrix[u] = 0
				usage_matrix[u] += 1
				total_usage += 1
			for i in range(self.model_size):
				stats['model_{}'.format(i)] = 0
			for key, value in usage_matrix.items():
				stats['model_{}'.format(key)] = value/total_usage if total_usage != 0 else 0
		return stats
		
	def add_to_statistics(self, id):
		self._model_usage_list.append(id)
		if len(self._model_usage_list) > flags.match_count_for_evaluation:
			self._model_usage_list.popleft() # remove old statistics
		
	def get_shared_keys(self):
		vars = []
		for agent in self.model_list:
			vars += agent.get_shared_keys()
		return vars
		
	def reset(self):
		self.step = 0
		self.agent_id = 0
		# Internal states
		self.internal_states = None if flags.share_internal_state else [None]*self.model_size
		# Count based exploration
		if flags.use_count_based_exploration_reward:
			self.hash_state_table = {}
			
	def initialize_new_batch(self):
		self.batch = ExperienceBatch(self.model_size)
		
	def estimate_value(self, agent_id, states, concats=None, internal_state=None):
		return self.get_model(agent_id).predict_value(states=states, concats=concats, internal_state=internal_state)
		
	def act(self, act_function, state, concat=None):
		agent_id = self.agent_id
		agent = self.get_model(agent_id)
		
		internal_state = self.internal_states if flags.share_internal_state else self.internal_states[agent_id]
		action_batch, value_batch, policy_batch, new_internal_state = agent.predict_action(states=[state], concats=[concat], internal_state=internal_state)
		if flags.share_internal_state:
			self.internal_states = new_internal_state
		else:
			self.internal_states[agent_id] = new_internal_state
		
		action, value, policy = action_batch[0], value_batch[0], policy_batch[0]
		new_state, extrinsic_reward, terminal = act_function(action)
		if flags.clip_reward:
			extrinsic_reward = np.clip(extrinsic_reward, flags.min_reward, flags.max_reward)
			
		intrinsic_reward = 0
		if flags.use_count_based_exploration_reward: # intrinsic reward
			intrinsic_reward += self.get_count_based_exploration_reward(new_state)

		total_reward = np.array([extrinsic_reward, intrinsic_reward])
		self.batch.add_action(agent_id=agent_id, state=state, concat=concat, action=action, policy=policy, reward=total_reward, value=value, internal_state=internal_state)
		# update step at the end of the action
		self.step += 1
		# return result
		return new_state, value, action, total_reward, terminal, policy

	def get_count_based_exploration_reward(self, new_state):
		if len(self.projection_train_set) < flags.projection_train_set_size:
			self.projection_train_set.append(new_state.flatten())
		if len(self.projection_train_set) == flags.projection_train_set_size:
			if self.projection is None:
				self.projection = random_projection.SparseRandomProjection(n_components=flags.exploration_hash_size if flags.exploration_hash_size > 0 else 'auto') # http://scikit-learn.org/stable/modules/random_projection.html
			self.projection.fit(self.projection_train_set)
			self.projection_train_set = [] # reset
		if self.projection is not None:
			state_projection = self.projection.transform([new_state.flatten()])[0] # project to smaller dimension
			state_hash = ''.join('1' if x > 0 else '0' for x in state_projection) # build binary locality-sensitive hash
			if state_hash not in self.hash_state_table:
				self.hash_state_table[state_hash] = 1
				if self.step > flags.steps_before_exploration_reward_bonus:
					return flags.exploration_bonus
		return 0
			
	def compute_discounted_cumulative_reward(self, batch):
		last_value = batch.bootstrap['value'] if 'value' in batch.bootstrap else 0.
		batch.compute_discounted_cumulative_reward(agents=self.agents_set, last_value=last_value, gamma=flags.gamma, lambd=flags.lambd)
		return batch
		
	def train(self, batch):
		# assert self.global_network is not None, 'Cannot train the global network.'
		internal_states = batch.internal_states
		states = batch.states
		concats = batch.concats
		actions = batch.actions
		policies = batch.policies
		values = batch.values
		rewards = batch.rewards
		dcr = batch.discounted_cumulative_rewards
		gae = batch.generalized_advantage_estimators
		batch_error = []
		for i in range(self.model_size):
			batch_size = len(states[i])
			if batch_size > 0:
				model = self.get_model(i)
				# reward prediction
				if model.predict_reward:
					sampled_batch = self.reward_prediction_buffer.sample()
					reward_prediction_states, reward_prediction_target = self.get_reward_prediction_tuple(sampled_batch)
				else:
					reward_prediction_states = None
					reward_prediction_target = None
				# train
				error, train_info = model.train(
					states=states[i], concats=concats[i],
					actions=actions[i], values=values[i],
					policies=policies[i],
					rewards=rewards[i],
					discounted_cumulative_rewards=dcr[i],
					generalized_advantage_estimators=gae[i],
					reward_prediction_states=reward_prediction_states,
					reward_prediction_target=reward_prediction_target,
					internal_state=internal_states[i][0]
				)
				batch_error.append(error)
				# loss statistics
				if flags.print_loss:
					for key, value in train_info.items():
						if key not in self._loss_list[i]:
							self._loss_list[i][key] = deque()
						self._loss_list[i][key].append(value)
						if len(self._loss_list[i][key]) > flags.match_count_for_evaluation: # remove old statistics
							self._loss_list[i][key].popleft()
		return batch_error
		
	def bootstrap(self, state, concat=None):
		agent_id = self.agent_id
		internal_state = self.internal_states if flags.share_internal_state else self.internal_states[agent_id]
		value_batch, _ = self.estimate_value(agent_id=agent_id, states=[state], concats=[concat], internal_state=internal_state)
		bootstrap = self.batch.bootstrap
		bootstrap['internal_state'] = internal_state
		bootstrap['agent_id'] = agent_id
		bootstrap['state'] = state
		bootstrap['concat'] = concat
		bootstrap['value'] = value_batch[0]
		
	def replay_value(self, batch): # replay values
		# replay values
		for (agent_id,pos) in batch.step_generator(self.agents_set):
			concat, state, internal_state = batch.get_action(['concats','states','internal_states'], agent_id, pos)
			value_batch, _ = self.estimate_value(agent_id=agent_id, states=[state], concats=[concat], internal_state=internal_state)
			batch.set_action({'values':value_batch[0]}, agent_id, pos)
		if 'value' in batch.bootstrap:
			bootstrap = batch.bootstrap
			agent_id = bootstrap['agent_id']
			value_batch, _ = self.estimate_value(agent_id=agent_id, states=[bootstrap['state']], concats=[bootstrap['concat']], internal_state=bootstrap['internal_state'])
			bootstrap['value'] = value_batch[0]
		return self.compute_discounted_cumulative_reward(batch)
		
	def add_to_reward_prediction_buffer(self, batch):
		batch_size = batch.get_size(self.agents_set)
		if batch_size <= 3:
			return
		tot_reward = np.sum(batch.get_cumulative_reward(self.agents_set))
		self.reward_prediction_buffer.put(batch=batch, type_id=1 if tot_reward != 0 else 0) # process batch only after sampling, for better perfomance
			
	def get_reward_prediction_tuple(self, batch):
		flat_states = []
		flat_rewards = []
		for agent in self.agents_set:
			flat_states.extend(batch.states[agent])
			flat_rewards.extend(batch.rewards[agent])
		states_count = len(flat_states)
		start_idx = np.random.randint(states_count-3)
		reward_prediction_states = [flat_states[start_idx+i] for i in range(3)]
		reward_prediction_target = np.zeros((1,3))
		
		target_reward = np.sum(flat_rewards[start_idx+3])
		if target_reward == 0:
			reward_prediction_target[0][0] = 1.0 # zero
		elif target_reward > 0:
			reward_prediction_target[0][1] = 1.0 # positive
		else:
			reward_prediction_target[0][2] = 1.0 # negative
		return reward_prediction_states, reward_prediction_target
			
	def add_to_replay_buffer(self, batch, batch_error):
		batch_size = batch.get_size(self.agents_set)
		if batch_size < 1:
			return
		batch_reward = batch.get_cumulative_reward(self.agents_set)[0]
		if batch_reward == 0 and flags.save_only_batches_with_reward:
			return
		if flags.replay_using_default_internal_state:
			batch.reset_internal_states()
		type_id = 1 if batch_reward > 0 else 0
		if flags.prioritized_replay:
			self.experience_buffer.put(batch=batch, priority=sum(batch_error)/len(batch_error), type_id=type_id)
		else:
			self.experience_buffer.put(batch=batch, type_id=type_id)
		
	def replay_experience(self):
		if not self.experience_buffer.has_atleast(flags.replay_start):
			return
		n = np.random.poisson(flags.replay_ratio)
		for _ in range(n):
			if flags.prioritized_replay:
				old_batch, batch_index, type_id = self.experience_buffer.sample()
			else:
				old_batch = self.experience_buffer.sample()
			old_batch_error = self.train(self.replay_value(old_batch) if flags.replay_value else old_batch)
			if flags.prioritized_replay:
				avg_error = sum(old_batch_error)/len(old_batch_error)
				self.experience_buffer.update_priority(batch_index, avg_error, type_id)
		
	def process_batch(self, global_step):
		batch = self.compute_discounted_cumulative_reward(self.batch)
		# reward prediction
		if flags.predict_reward:
			self.add_to_reward_prediction_buffer(batch) # do it before training, this way there will be at least one batch in the reward_prediction_buffer
			if self.reward_prediction_buffer.is_empty():
				return # cannot train without reward prediction, wait until reward_prediction_buffer is not empty
		# train
		batch_error = self.train(batch)
		# experience replay (after training!)
		if flags.replay_ratio > 0 and global_step > flags.replay_step:
			self.replay_experience()
			self.add_to_replay_buffer(batch, batch_error)