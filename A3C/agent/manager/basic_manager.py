# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import tensorflow as tf
import numpy as np
from agent.network import *
from agent.experience_buffer import Buffer
from agent.batch import ExperienceBatch, RewardPredictionBatch

import options
flags = options.get()

class BasicManager(object):
	#__slots__ = ('model_list','_model_usage_list','experience_buffer','reward_prediction_buffer','global_network','model_size','sync_list','batch',
	#				'training','session','id','device','clip',
	#				'global_step','learning_rate','gradient_optimizer',
	#				'agent_id','lstm_state')
	
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
			self.experience_buffer = Buffer(size=flags.replay_buffer_size)
		if flags.predict_reward:
			self.reward_prediction_buffer = Buffer(size=flags.reward_prediction_buffer_size)
	# Bind optimizer to global
		if not self.is_global_network():
			self.bind_to_global(self.global_network)
	# Statistics
		self._model_usage_list = deque()
		if flags.print_loss:
			self._loss_list = [{"total_loss": deque(), "policy_loss": deque(), "value_loss": deque()} for _ in range(self.model_size)]
			
	def is_global_network(self):
		return self.global_network is None
			
	def set_model_size(self):
		self.model_size = 1
			
	def build_agents(self, state_shape, action_shape, concat_size):
		agent=eval('{}_Network'.format(flags.network))(
			session=self.session,
			id='{0}_{1}'.format(self.id, 0),
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
			local_agent.minimize_local(optimizer=global_network.gradient_optimizer[i], global_step=global_network.global_step[i], global_var_list=global_agent.get_shared_keys())
			self.sync_list.append(local_agent.bind_sync(global_agent)) # for syncing local network with global one

	def get_model(self, id):
		return self.model_list[id]
		
	def get_statistics(self):
		stats = {}
		# build loss statistics
		if flags.print_loss:
			for i in range(self.model_size):
				if len(self._loss_list[i]["total_loss"]) > 0:
					stats['loss_total{}'.format(i)] = sum(self._loss_list[i]["total_loss"])/len(self._loss_list[i]["total_loss"])
					stats['loss_policy{}'.format(i)] = sum(self._loss_list[i]["policy_loss"])/len(self._loss_list[i]["policy_loss"])
					stats['loss_value{}'.format(i)] = sum(self._loss_list[i]["value_loss"])/len(self._loss_list[i]["value_loss"])
				else:
					stats['loss_total{}'.format(i)] = 0
					stats['loss_policy{}'.format(i)] = 0
					stats['loss_value{}'.format(i)] = 0
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
		self.agent_id = 0
		if flags.share_lstm_states:
			self.lstm_state = None
		else:
			self.lstm_state = [None for _ in range(self.model_size)] # do not mix lstm states from different agents
			
	def initialize_new_batch(self):
		self.batch = ExperienceBatch(self.model_size)
		
	def estimate_value(self, agent_id, states, concats=None, lstm_state=None):
		return self.get_model(agent_id).predict_value(states=states, concats=concats, lstm_state=lstm_state)
		
	def act(self, act_function, state, concat=None):
		agent_id = self.agent_id
		agent = self.get_model(agent_id)
		if flags.share_lstm_states:		
			lstm_state = self.lstm_state
			action_batch, value_batch, policy_batch, self.lstm_state = agent.predict_action(states=[state], concats=[concat], lstm_state=lstm_state)
		else:
			lstm_state = self.lstm_state[agent_id]
			action_batch, value_batch, policy_batch, self.lstm_state[agent_id] = agent.predict_action(states=[state], concats=[concat], lstm_state=lstm_state)
		action, value, policy = action_batch[0], value_batch[0], policy_batch[0]
		new_state, reward, terminal = act_function(action)
		if flags.clip_reward:
			reward = np.clip(reward, flags.min_reward, flags.max_reward)

		self.batch.add_agent_action(agent_id, state, concat, action, policy, reward, value, lstm_state)
		return new_state, value, action, reward, terminal, policy
					
	def compute_cumulative_reward(self, batch):
		# prepare batch
		for i in range(self.model_size):
			batch.discounted_cumulative_rewards[i]=deque()
			batch.generalized_advantage_estimators[i]=deque()
		# bootstrap
		discounted_cumulative_reward = 0.0
		generalized_advantage_estimator = 0.0
		if 'value' in batch.bootstrap:
			discounted_cumulative_reward = batch.bootstrap['value']
		# compute cumulative reward and advantage
		last_value = discounted_cumulative_reward
		batch_size = batch.size
		for t in range(batch_size):
			index = -t-1
			value, reward = batch.get_step_action(['values','rewards'], index)
			discounted_cumulative_reward = reward + flags.gamma * discounted_cumulative_reward
			generalized_advantage_estimator = reward + flags.gamma * last_value - value + flags.gamma*flags.lambd*generalized_advantage_estimator
			last_value = value
			batch.set_step_action({'discounted_cumulative_rewards':discounted_cumulative_reward, 'generalized_advantage_estimators':generalized_advantage_estimator}, index)
		return batch
		
	def train(self, batch):
		# assert self.global_network is not None, 'you are trying to train the global network'
		states = batch.states
		concats = batch.concats
		actions = batch.actions
		policies = batch.policies
		values = batch.values
		rewards = batch.rewards
		dcr = batch.discounted_cumulative_rewards
		gae = batch.generalized_advantage_estimators
		lstm_states = batch.lstm_states
		for i in range(self.model_size):
			batch_size = len(states[i])
			if batch_size > 0:
				model = self.get_model(i)
				# reward prediction
				if model.predict_reward:
					rp_batch = self.reward_prediction_buffer.get()
					rp_states, rp_target = rp_batch.states, rp_batch.target
				else:
					rp_states = None
					rp_target = None
				# train
				total_loss, policy_loss, value_loss = model.train(
					states=states[i], concats=concats[i],
					actions=actions[i], values=values[i],
					policies=policies[i],
					rewards=rewards[i],
					discounted_cumulative_rewards=dcr[i],
					generalized_advantage_estimators=gae[i],
					lstm_state=lstm_states[i][0],
					reward_prediction_states=rp_states,
					reward_prediction_target=rp_target
				)
				# loss statistics
				if flags.print_loss:
					self._loss_list[i]["total_loss"].append(total_loss)
					self._loss_list[i]["policy_loss"].append(policy_loss)
					self._loss_list[i]["value_loss"].append(value_loss)
					if len(self._loss_list[i]["total_loss"]) > flags.match_count_for_evaluation: # remove old statistics
						self._loss_list[i]["total_loss"].popleft()
						self._loss_list[i]["policy_loss"].popleft()
						self._loss_list[i]["value_loss"].popleft()
				
	def bootstrap(self, state, concat=None):
		agent_id = self.agent_id
		if flags.share_lstm_states:
			lstm_state = self.lstm_state
		else:
			lstm_state = self.lstm_state[agent_id]
		value_batch, _ = self.estimate_value(agent_id=agent_id, states=[state], concats=[concat], lstm_state=lstm_state)
		bootstrap = self.batch.bootstrap
		bootstrap['lstm_state'] = lstm_state
		bootstrap['agent_id'] = agent_id
		bootstrap['state'] = state
		bootstrap['concat'] = concat
		bootstrap['value'] = value_batch[0]
		
	def replay_value(self, batch): # replay values and lstm states
		if flags.share_lstm_states:
			lstm_state = batch.get_step_action('lstm_states', 0)
		else:
			lstm_state = [batch.lstm_states[i][0] if len(batch.lstm_states[i])>0 else None for i in range(self.model_size)]
		for i in range(batch.size):
			concat, state = batch.get_step_action(['concats','states'], i)
			agent_id, _ = batch.get_agent_and_pos(i)
			if flags.share_lstm_states:
				new_values, new_lstm_state = self.estimate_value(agent_id=agent_id, states=[state], concats=[concat], lstm_state=lstm_state)
				batch.set_step_action({'lstm_states':lstm_state,'values':new_values[0]}, i)
				lstm_state = new_lstm_state
			else:
				new_values, new_lstm_state = self.estimate_value(agent_id=agent_id, states=[state], concats=[concat], lstm_state=lstm_state[agent_id])
				batch.set_step_action({'lstm_states':lstm_state[agent_id],'values':new_values[0]}, i)
				lstm_state[agent_id] = new_lstm_state
		if 'value' in batch.bootstrap:
			bootstrap = batch.bootstrap
			agent_id = bootstrap['agent_id']
			if flags.share_lstm_states:
				values, _ = self.estimate_value(agent_id=agent_id, states=[bootstrap['state']], concats=[bootstrap['concat']], lstm_state=lstm_state)
			else:
				values, _ = self.estimate_value(agent_id=agent_id, states=[bootstrap['state']], concats=[bootstrap['concat']], lstm_state=lstm_state[agent_id])
			bootstrap['value'] = values[0]
		return self.compute_cumulative_reward(batch)
		
	def should_save_batch(self, batch):
		# Prioritize smaller batches because they have terminated prematurely, thus they are probably more important and also faster to process
		return np.random.randint(batch.size) == 0
		
	def add_to_reward_prediction_buffer(self, batch):
		batch_size = batch.size
		if batch_size <= 3:
			return
		should_save_batch = self.should_save_batch(batch)
		for i in range(3, batch_size):
			reward = batch.get_step_action('rewards', i)
			type_id = 1 if reward > 0 else 0
			if should_save_batch or not self.reward_prediction_buffer.id_is_full(type_id):
				states = [batch.get_step_action('states', j)  for j in range(i-3, i)]
				target = np.zeros((1,3))
				if reward == 0:
					target[0][0] = 1 # zero
				elif reward > 0:
					target[0][1] = 1 # positive
				else:
					target[0][2] = 1 # negative
				self.reward_prediction_buffer.put(batch=RewardPredictionBatch(states, target), type_id=type_id)  # target must have the same batch size of states to avoid tensorflow memory leaks
			
	def add_to_replay_buffer(self, batch):
		batch_size = batch.size
		if batch_size <= 1:
			return
		batch_reward = batch.total_reward
		if batch_reward == 0 and flags.save_only_batches_with_reward:
			return
		type_id = 1 if batch_reward > 0 else 0
		if not self.experience_buffer.id_is_full(type_id) or self.should_save_batch(batch):
			self.experience_buffer.put(batch, type_id)
		# self.experience_buffer.put(batch, 1 if batch_reward > 0 else 0)
		
	def process_batch(self, global_step):
		batch = self.compute_cumulative_reward(self.batch)
		# reward prediction
		if flags.predict_reward:
			self.add_to_reward_prediction_buffer(batch) # do it before training, this way there will be at least one batch in the reward_prediction_buffer
			if self.reward_prediction_buffer.is_empty():
				return # cannot train without reward prediction, wait until reward_prediction_buffer is not empty
		# train
		self.train(batch)
		# experience replay
		if flags.replay_ratio > 0 and global_step > flags.replay_step:
			if self.experience_buffer.has_atleast(flags.replay_start):
				n = np.random.poisson(flags.replay_ratio)
				for _ in range(n):
					old_batch = self.experience_buffer.get()
					self.train(self.replay_value(old_batch) if flags.replay_value else old_batch)
			self.add_to_replay_buffer(batch)