# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import options
flags = options.get()

from model.experience_buffer import Buffer
from model.loss.policy_loss import PolicyLoss
from model.loss.value_loss import ValueLoss

class BaseAC_Network(object):
	def __init__(self, session, id, state_shape, policy_size, entropy_beta, clip, device, predict_reward, concat_size=0):
		# learning rate stuff
		self.train_count = 0
		self.entropy_beta = entropy_beta
		self.clip = clip
		self.predict_reward = predict_reward
		if self.predict_reward:
			self.reward_prediction_buffer = Buffer(size=flags.reward_prediction_buffer_size)
		# initialize
		self._session = session
		self._id = id # model id
		self._device = device # gpu or cpu
		self._policy_size = policy_size # the dimension of the policy vector
		self._concat_size = concat_size # the size of the vector concatenated with the CNN output before entering the LSTM
		self._state_shape = state_shape # the shape of the input
		# lstm units
		self._lstm_units = 64 # the number of units of the LSTM
		# create the whole A3C network
		self._create_network()
	
	def _create_network(self):
		scope_name = "net_{0}".format(self._id)
		with tf.device(self._device), tf.variable_scope(scope_name) as scope:
			self._build_base()
			if self.predict_reward:
				self._build_reward_prediction()
			
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
		
	def _build_reward_prediction(self):
		self._reward_prediction_states = tf.placeholder(tf.float32, np.concatenate([[None], self._state_shape], 0))
		output = self._convolutive_layers(self._reward_prediction_states, reuse=True)
		output = tf.layers.flatten(output)
		output = tf.layers.dense(inputs=output, units=3, activation=None, kernel_initializer=tf.initializers.variance_scaling)
		self._reward_prediction = tf.nn.softmax(output)
		
	def _build_base(self):
		self._input = tf.placeholder(tf.float32, np.concatenate([[None], self._state_shape], 0))
		self._concat = tf.placeholder(tf.float32, [None, self._concat_size])
		# Convolutive layers
		conv_output = self._convolutive_layers(self._input)
		# LSTM layers
		self._lstm_outputs, self._lstm_state = self._lstm_layers(conv_output)
		# Policy layers
		self._policy = self._policy_layers(self._lstm_outputs)
		# Value layers
		self._value	= self._value_layers(self._lstm_outputs)
		
	def _convolutive_layers(self, input, reuse=False):
		with tf.variable_scope("base_conv{0}".format(self._id), reuse=reuse) as scope:
			# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=16, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.relu) # xavier initializer
			# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=32, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.relu) # xavier initializer
			input = tf.layers.conv2d( inputs=input, filters=16, kernel_size=(3,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )
			input = tf.layers.conv2d( inputs=input, filters=8, kernel_size=(3,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )
			return input
	
	def _lstm_layers(self, input, reuse=False):
		self._initial_lstm_state0 = tf.placeholder(tf.float32, [1, self._lstm_units])
		self._initial_lstm_state1 = tf.placeholder(tf.float32, [1, self._lstm_units])
		self._initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self._initial_lstm_state0, self._initial_lstm_state1)
		self.reset()
		with tf.variable_scope("base_lstm{0}".format(self._id), reuse=reuse) as scope:
		
			input = tf.layers.flatten(input) # input shape: (batch,w*h*depth)
			# input = tf.contrib.model_pruning.masked_fully_connected(inputs=input, num_outputs=self._lstm_units, activation_fn=tf.nn.relu) # xavier initializer
			input = tf.layers.dense( inputs=input, units=self._lstm_units, activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )
			step_size = tf.shape(input)[:1] # (unroll_step, 256)
			if self._concat_size > 0:
				input = tf.concat([input, self._concat], 1) # (unroll_step, 256+policy_size+1)
				input = tf.reshape(input, [1, -1, self._lstm_units+self._concat_size]) # (1, unroll_step, 256+policy_size+1)
			else:
				input = tf.reshape(input, [1, -1, self._lstm_units]) # (1, unroll_step, 256)

			# self._lstm_cell = tf.contrib.model_pruning.MaskedBasicLSTMCell(num_units=self._lstm_units, forget_bias=1.0, state_is_tuple=True, activation=None)
			self._lstm_cell = tf.contrib.rnn.BasicLSTMCell(self._lstm_units, state_is_tuple=True)
			lstm_outputs, lstm_state = tf.nn.dynamic_rnn(self._lstm_cell, input, initial_state = self._initial_lstm_state, sequence_length = step_size, time_major = False, scope = scope)
			# Dropout: https://www.nature.com/articles/s41586-018-0102-6
			lstm_outputs = tf.layers.dropout(inputs=lstm_outputs, rate=0.5)
			
			lstm_outputs = tf.reshape(lstm_outputs, [-1,self._lstm_units]) #(1,unroll_step,256) for back prop, (1,1,256) for forward prop.
			return lstm_outputs, lstm_state

	def _policy_layers(self, input, reuse=False): # Policy (output)
		with tf.variable_scope("base_policy{0}".format(self._id), reuse=reuse) as scope:
			input = tf.layers.dense( inputs=input, units=self._policy_size, activation=None, kernel_initializer=tf.initializers.variance_scaling )
			return tf.nn.softmax(input)

	def _value_layers(self, input, reuse=False): # Value (output)
		with tf.variable_scope("base_value{0}".format(self._id), reuse=reuse) as scope:
			input = tf.layers.dense( inputs=input, units=1, activation=None, kernel_initializer=tf.initializers.variance_scaling )
			return tf.reshape(input, [-1]) # flatten output

	def prepare_loss(self):
		with tf.device(self._device):
			# Batch state values
			self.old_value_batch = tf.placeholder(tf.float32, [None])
			# Batch policies
			self.old_policy_batch = tf.placeholder(tf.float32, [None, self._policy_size])
			# Taken action (input for policy)
			self.action_batch = tf.placeholder(tf.float32, [None, self._policy_size])
			# Advantage (R-V) (input for policy)
			self.advantage_batch = tf.placeholder(tf.float32, [None])
			# R (input for value target)
			self.cumulative_reward_batch = tf.placeholder(tf.float32, [None])
			# Loss
			self.policy_loss = PolicyLoss(self.clip, self._policy, self.old_policy_batch, self._value, self.old_value_batch, self.action_batch, self.advantage_batch, self.cumulative_reward_batch, self.entropy_beta)
			self.value_loss = ValueLoss(self.clip, self._policy, self.old_policy_batch, self._value, self.old_value_batch, self.action_batch, self.advantage_batch, self.cumulative_reward_batch, self.entropy_beta)
			
			self.total_loss = self.policy_loss.get() + flags.value_coefficient*self.value_loss.get()
			if self.predict_reward:
				self.total_loss += self.reward_prediction_loss()
			
	def reward_prediction_loss(self):
		# reward prediction target. one hot vector
		self.reward_prediction_target = tf.placeholder("float", [1,3])
		# Reward prediction loss (output)
		clipped_rp = tf.clip_by_value(self._reward_prediction, 1e-20, 1.0)
		return -tf.reduce_sum(self.reward_prediction_target * tf.log(clipped_rp))

	def reset(self):
		# reset lstm state output
		self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, self._lstm_units]), np.zeros([1, self._lstm_units]))

	def run_policy_and_value(self, state, concat=None):
		# This run_policy_and_value() is used when forward propagating.
		# so the step size is 1.
		feed_dict = { 
				self._input : state, 
				self._initial_lstm_state0 : self.lstm_state_out[0], 
				self._initial_lstm_state1 : self.lstm_state_out[1]
			}
		if self._concat_size > 0:
			feed_dict.update( { self._concat : concat } )
		pi_out, v_out, self.lstm_state_out = self._session.run( [self._policy, self._value, self._lstm_state], feed_dict = feed_dict )
		# pi_out: (1,3), v_out: (1)
		return (pi_out[0], v_out[0])
		
	def sync(self, sync):
		self._session.run(sync)
		
	def run_value(self, state, concat=None):
		# This run_value() is used for calculating V for bootstrapping at the 
		# end of MAX_BATCH_SIZE time step sequence.
		# When next sequence starts, V will be calculated again with the same state using updated network weights,
		# so we don't update LSTM state here.
		feed_dict = {
				self._input : state, 
				self._initial_lstm_state0 : self.lstm_state_out[0], 
				self._initial_lstm_state1 : self.lstm_state_out[1] 
			}
		if self._concat_size > 0:
			feed_dict.update( { self._concat : concat } )
		v_out, _ = self._session.run( [self._value, self._lstm_state], feed_dict = feed_dict )
		return v_out[0]
		
	def bind_sync(self, src_network, name=None):
		src_vars = src_network.get_vars()
		dst_vars = self.get_vars()
		sync_ops = []
		with tf.device(self._device):
			with tf.name_scope(name, "A3CModel{0}".format(self._id),[]) as name:
				for(src_var, dst_var) in zip(src_vars, dst_vars):
					sync_op = tf.assign(dst_var, src_var)
					sync_ops.append(sync_op)
				return tf.group(*sync_ops, name=name)
				
	def minimize_local(self, optimizer, global_step, global_var_list):
		"""
		minimize loss and apply gradients to global vars.
		"""
		loss = self.total_loss
		local_var_list = self.get_vars()
		with tf.device(self._device):
			var_refs = [v._ref() for v in local_var_list]
			local_gradients = tf.gradients(
				loss, var_refs,
				gate_gradients=False,
				aggregation_method=None,
				colocate_gradients_with_ops=False)
			if flags.grad_norm_clip > 0:
				local_gradients, _ = tf.clip_by_global_norm(local_gradients, flags.grad_norm_clip)
			grads_and_vars = list(zip(local_gradients, global_var_list))
			self.apply_gradient = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
				
	def get_vars(self):
		return self.variables # get model variables
		
	def get_action_vector(self, action): # transform action into a 1-hot-vector
		hot_vector = np.zeros([self._policy_size])
		hot_vector[action] = 1.0
		return hot_vector
		
	def _lstm_state_placeholder(self):
		return tf.placeholder(tf.float32, [1, self._lstm_units])
				
	def state_target_reward_prediction(self, states, rewards):
		if len(states) == 1:
			states = states + states
			rewards = rewards + rewards
		length = min(3, len(states)-1)
			
		start_idx = np.random.randint(len(states)-length)
		reward_prediction_states = [states[start_idx+i] for i in range(length)]
		reward_prediction_target = [0.0, 0.0, 0.0]
		
		r = rewards[start_idx+length]
		if r == 0:
			reward_prediction_target[0] = 1.0 # zero
		elif r > 0:
			reward_prediction_target[1] = 1.0 # positive
		else:
			reward_prediction_target[2] = 1.0 # negative
		return reward_prediction_states, [reward_prediction_target]
		
	def build_feed(self, states, actions, rewards, values, policies, discounted_cumulative_rewards, generalized_advantage_estimators, lstm_states, concat):
		values = np.reshape(values,[-1])
		if flags.use_GAE: # Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).
			advantages = np.reshape(generalized_advantage_estimators,[-1])
			cumulative_rewards = advantages + values
			# advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # batch normalization
		else:
			cumulative_rewards = np.reshape(discounted_cumulative_rewards,[-1])
			advantages = cumulative_rewards - values			
		feed_dict={
					self._input: states,
					self.action_batch: actions,
					self.cumulative_reward_batch: cumulative_rewards,
					self.advantage_batch: advantages,
					self.old_value_batch: values,
					self.old_policy_batch: policies,
					self._initial_lstm_state: lstm_states[0]
				}
		if self._concat_size > 0:
			feed_dict.update( {self._concat : concat} )
		if self.predict_reward:
			self.reward_prediction_buffer.put(batch={"states":states, "rewards":rewards}, type=1 if sum(rewards) != 0 else 0)
			(rpb,_) = self.reward_prediction_buffer.get()
			reward_prediction_states, reward_prediction_target = self.state_target_reward_prediction(rpb["states"], rpb["rewards"])
			feed_dict.update( {
				self._reward_prediction_states: reward_prediction_states,
				self.reward_prediction_target: reward_prediction_target
			} )
		return feed_dict
				
	def train(self, states, actions, rewards, values, policies, discounted_cumulative_rewards, generalized_advantage_estimators, lstm_states, concat=None):
		self.train_count += len(states)
		feed_dict = self.build_feed(states, actions, rewards, values, policies, discounted_cumulative_rewards, generalized_advantage_estimators, lstm_states, concat)
		self._session.run(self.apply_gradient, feed_dict) # Calculate gradients and copy them to global network