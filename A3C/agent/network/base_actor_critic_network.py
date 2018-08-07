# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import options
flags = options.get()

from agent.loss.policy_loss import PolicyLoss
from agent.loss.value_loss import ValueLoss

class BaseAC_Network(object):
	def __init__(self, session, id, state_shape, action_shape, entropy_beta, clip, device, predict_reward, concat_size=0, training=True):
		# learning rate stuff
		self.train_count = 0
		self.entropy_beta = entropy_beta
		self.clip = clip
		self.predict_reward = predict_reward
		# initialize
		self._training = training
		self._session = session
		self._id = id # model id
		self._device = device # gpu or cpu
		self.policy_length = action_shape[0] # the dimension of the policy vector
		self.policy_depth = action_shape[1] if len(action_shape) > 1 else 0 # dimension of the softmax: 0 for none, 1 if you want a single softmax for the whole policy, 2 or more if you want a softmax (with dimension self.policy_depth) for any policy element
		self._concat_size = concat_size # the size of the vector concatenated with the CNN output before entering the LSTM
		self._state_shape = state_shape # the shape of the input
		# lstm units
		self._lstm_units = 64 # the number of units of the LSTM
		# create the whole A3C network
		self._create_network()
		
	def _lstm_state_placeholder(self, name=None):
		return tf.placeholder(dtype=tf.float32, shape=[1, self._lstm_units], name=name)
		
	def _state_placeholder(self, name=None):
		return tf.placeholder(dtype=tf.float32, shape=np.concatenate([[None], self._state_shape], 0), name=name)
		
	def _policy_placeholder(self, name=None):
		if self.policy_depth < 2:
			return tf.placeholder(dtype=tf.float32, shape=[None,self.policy_length], name=name)
		else:
			return tf.placeholder(dtype=tf.float32, shape=[None,self.policy_length,self.policy_depth], name=name)
		
	def _value_placeholder(self, name=None):
		return tf.placeholder(dtype=tf.float32, shape=[None,1], name=name)
		
	def _concat_placeholder(self, name=None):
		return tf.placeholder(dtype=tf.float32, shape=[None, self._concat_size], name=name)
		
	def _reward_prediction_target_placeholder(self, name=None):
		return tf.placeholder(dtype=tf.float32, shape=[1,3], name=name)
	
	def _create_network(self):
		scope_name = "net_{0}".format(self._id)
		with tf.device(self._device), tf.variable_scope(scope_name) as scope:
			self._build_base()
			if self.predict_reward:
				self._build_reward_prediction()
		self.train_keys = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
		self.update_keys = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope_name) # for batch normalization
		
	def _build_base(self):
		print( "Building network {}".format(self._id) )
		# [Input]
		self.state_batch = self._state_placeholder("state")
		print( "    [{}]Input shape: {}".format(self._id, self.state_batch.get_shape()) )
		# [Concatenation]
		self._concat = self._concat_placeholder("concat")
		print( "    [{}]Concatenation shape: {}".format(self._id, self._concat.get_shape()) )
		# [CNN tower]
		tower = self._convolutive_layers(self.state_batch)
		print( "    [{}]Tower shape: {}".format(self._id, tower.get_shape()) )
		# [LSTM]
		lstm, self._lstm_state = self._lstm_layers(tower, self._concat)
		print( "    [{}]LSTM shape: {}".format(self._id, lstm.get_shape()) )
		# [Policy]
		self.policy_batch = self._policy_layers(lstm)
		print( "    [{}]Policy shape: {}".format(self._id, self.policy_batch.get_shape()) )
		# [Value]
		self.value_batch = self._value_layers(lstm)
		print( "    [{}]Value shape: {}".format(self._id, self.value_batch.get_shape()) )
		
	def _build_reward_prediction(self):
		self._reward_prediction_states = self._state_placeholder("reward_prediction_state")
		output = self._convolutive_layers(self._reward_prediction_states, reuse=True)
		output = tf.layers.flatten(output)
		self.reward_prediction_logits = tf.layers.dense(inputs=output, units=3, activation=None, kernel_initializer=tf.initializers.variance_scaling)
		
	def _convolutive_layers(self, input, reuse=False):
		with tf.variable_scope("base_conv{0}".format(self._id), reuse=reuse) as scope:
			# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=16, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.relu) # xavier initializer
			# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=32, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.relu) # xavier initializer
			input = tf.layers.conv2d( inputs=input, filters=16, kernel_size=(3,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )
			input = tf.layers.conv2d( inputs=input, filters=8, kernel_size=(3,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )
			input = tf.layers.batch_normalization(input, training=self._training)
			return input
	
	def _lstm_layers(self, input, concat, reuse=False):
		with tf.variable_scope("base_lstm{0}".format(self._id), reuse=reuse) as scope:
			input = tf.layers.flatten(input) # shape: (batch,w*h*depth)
			# input = tf.contrib.model_pruning.masked_fully_connected(inputs=input, num_outputs=self._lstm_units, activation_fn=tf.nn.relu) # xavier initializer
			input = tf.layers.dense( inputs=input, units=self._lstm_units, activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )
			step_size = tf.shape(input)[:1] # shape: (batch)
			if self._concat_size > 0:
				input = tf.concat([input, concat], 1) # shape: (batch, concat_size+lstm_units)
				input = tf.reshape(input, [1, -1, self._lstm_units+self._concat_size]) # shape: (1, batch, concat_size+lstm_units)
			else:
				input = tf.reshape(input, [1, -1, self._lstm_units]) # shape: (1, batch, lstm_units)

			# self._lstm_cell = tf.contrib.model_pruning.MaskedBasicLSTMCell(num_units=self._lstm_units, forget_bias=1.0, state_is_tuple=True, activation=None)
			self._lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self._lstm_units)
			self._initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self._lstm_state_placeholder("lstm_tuple_1"), self._lstm_state_placeholder("lstm_tuple_2"))
			self._empty_lstm_state = (np.zeros([1,self._lstm_units]),np.zeros([1,self._lstm_units]))
			lstm_outputs, lstm_state = tf.nn.dynamic_rnn(cell=self._lstm_cell, inputs=input, initial_state=self._initial_lstm_state, sequence_length=step_size, time_major = False, scope = scope)
			
			# Dropout: https://www.nature.com/articles/s41586-018-0102-6
			lstm_outputs = tf.layers.dropout(inputs=lstm_outputs, rate=0.5)
			
			lstm_outputs = tf.reshape(lstm_outputs, [-1,self._lstm_units]) # shape: (batch, lstm_units)
			return lstm_outputs, lstm_state
			
	def _policy_layers(self, input, reuse=False): # Policy (output)
		with tf.variable_scope("base_policy{0}".format(self._id), reuse=reuse) as scope:
			if self.policy_depth < 2:
				input = tf.layers.dense(inputs=input, units=self.policy_length, activation=None, kernel_initializer=tf.initializers.variance_scaling)
				logits = input
				output = tf.nn.softmax(logits) if self.policy_depth > 0 else logits
			else:
				policy = []
				for i in range(self.policy_depth):
					p = tf.layers.dense(inputs=input, units=self.policy_length, activation=None, kernel_initializer=tf.initializers.variance_scaling)
					p = tf.expand_dims(p, axis=-1)
					policy.append(p)
				logits = tf.concat(policy, -1)
				output = tf.contrib.layers.softmax(logits)
			self.action_batch = self._policy_placeholder("action")
			self.cross_entropy_batch = self.get_cross_entropy(labels=self.action_batch, logits=logits, softmax=self.policy_depth > 0)
			self.entropy_batch = self.get_entropy(logits=logits, softmax=self.policy_depth > 0)
			return output

	def _value_layers(self, input, reuse=False): # Value (output)
		with tf.variable_scope("base_value{0}".format(self._id), reuse=reuse) as scope:
			input = tf.layers.dense( inputs=input, units=1, activation=None, kernel_initializer=tf.initializers.variance_scaling )
			return tf.reshape(input, [-1]) # flatten output
		
	def get_cross_entropy(self, labels, logits, softmax):
		if softmax: # discrete control
			return tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
			# The following alternative is: (i) less numerically stable (since the softmax may compute much larger values) and (ii) less efficient (since some redundant computation would happen in the backprop)
			# log_batch = tf.log(tf.clip_by_value(logits, 1e-8, 1.0)) # Avoid NaN with clipping when value in tensor becomes zero
			# return tf.reduce_sum(tf.multiply(log_batch, labels), reduction_indices=1)
		# else continuous control
		bn = tf.layers.BatchNormalization(trainable=self._training)
		batch_layer = bn.apply(logits, training=self._training)
		mean = bn.moving_mean
		std = tf.sqrt(bn.moving_variance)
		return 0.5 * tf.reduce_sum(tf.square((labels - mean) / std), axis=-1) \
			   + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(labels)[-1]) \
			   + tf.reduce_sum(tf.log(std), axis=-1)
			
	def get_entropy(self, logits, softmax):
		if softmax: # discrete control
			a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
			ea0 = tf.exp(a0)
			z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
			p0 = ea0 / z0
			return tf.reduce_sum( tf.multiply(p0, (tf.log(z0) - a0)), axis=-1)
		# else continuous control
		bn = tf.layers.BatchNormalization(trainable=True)
		batch_layer = bn.apply(logits, training=True)
		std = tf.sqrt(bn.moving_variance)
		return tf.reduce_sum(tf.log(std) + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
			
	def prepare_loss(self):
		with tf.device(self._device):
			# Batch placeholders
			self.old_policy_batch = self._policy_placeholder("old_policy")
			self.old_value_batch = self._value_placeholder("old_value")
			self.advantage_batch = self._value_placeholder("advantage")
			self.cumulative_reward_batch = self._value_placeholder("cumulative_reward")
			self.old_cross_entropy_batch = self._value_placeholder("old_cross_entropy")
			# Build losses
			self.policy_loss = PolicyLoss(
				cliprange=self.clip,
				policy=self.policy_batch,
				old_policy=self.old_policy_batch,
				cross_entropy=self.cross_entropy_batch,
				old_cross_entropy=self.old_cross_entropy_batch,
				advantage=self.advantage_batch,
				entropy=self.entropy_batch,
				entropy_beta=self.entropy_beta
			)
			self.value_loss = ValueLoss(
				cliprange=self.clip, 
				value=self.value_batch, 
				old_value=self.old_value_batch, 
				reward=self.cumulative_reward_batch
			)
			# Compute total loss
			self.total_loss = self.policy_loss.get() + flags.value_coefficient*self.value_loss.get()
			if self.predict_reward:
				self.total_loss += self.reward_prediction_loss()
			
	def reward_prediction_loss(self):
		# reward prediction target. one hot vector
		self.reward_prediction_labels = self._reward_prediction_target_placeholder("reward_prediction_target")
		# Reward prediction loss (output)
		return -self.get_cross_entropy(labels=self.reward_prediction_labels, logits=self.reward_prediction_logits, softmax=True)

	def bind_sync(self, src_network):
		src_vars = src_network.get_vars()
		dst_vars = self.get_vars()
		sync_ops = []
		with tf.device(self._device):
			for(src_var, dst_var) in zip(src_vars, dst_vars):
				sync_op = tf.assign(ref=dst_var, value=src_var, use_locking=True)
				sync_ops.append(sync_op)
			return tf.group(*sync_ops)
				
	def sync(self, sync):
		self._session.run(fetches=sync, options=tf.RunOptions.NO_TRACE)
				
	def minimize_local(self, optimizer, global_step, global_var_list):
		"""
		minimize loss and apply gradients to global vars.
		"""
		with tf.device(self._device) and tf.control_dependencies(self.update_keys): # control_dependencies is for batch normalization
			loss = self.total_loss
			local_var_list = self.get_vars()
			var_refs = [v._ref() for v in local_var_list]
			local_gradients = tf.gradients(
				loss, var_refs,
				gate_gradients=False,
				aggregation_method=None,
				colocate_gradients_with_ops=False)
			if flags.grad_norm_clip > 0:
				local_gradients, _ = tf.clip_by_global_norm(local_gradients, flags.grad_norm_clip)
			grads_and_vars = list(zip(local_gradients, global_var_list))
			self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
				
	def get_vars(self):
		return self.train_keys # get model variables
		
	def get_action_vector(self, action=None): # transform action into a 1-hot-vector
		if type(action) not in [list,tuple]:
			action = [action]
		if self.policy_depth < 2:
			hot_vector = np.zeros([self.policy_length])
			for i in range(len(action)):
				hot_vector[action[i]] = 1
			return hot_vector
		else:
			hot_vector = np.zeros([self.policy_length, self.policy_depth])
			for i in range(self.policy_length):
				hot_vector[i][action[i]] = 1
			return hot_vector
		
	def run_cross_entropy(self, actions, states, concats=None, lstm_state=None):
		feed_dict = { 
				self.action_batch : actions,
				self.state_batch : states,
				self._initial_lstm_state: lstm_state if lstm_state is not None else self._empty_lstm_state
			}
		if self._concat_size > 0:
			feed_dict.update( { self._concat : concats } )
		entropy_batch, cross_entropy_batch, _, _, _ = self._session.run(fetches=[self.entropy_batch, self.cross_entropy_batch, self.policy_batch, self.value_batch, self._lstm_state], feed_dict=feed_dict, options=tf.RunOptions.NO_TRACE)
		return cross_entropy_batch, entropy_batch
		
	def run_policy_and_value(self, states, concats=None, lstm_state=None):
		feed_dict = { 
				self.state_batch : states,
				self._initial_lstm_state: lstm_state if lstm_state is not None else self._empty_lstm_state
			}
		if self._concat_size > 0:
			feed_dict.update( { self._concat : concats } )
		# return policies, values, lstm_state
		return self._session.run(fetches=[self.policy_batch, self.value_batch, self._lstm_state], feed_dict=feed_dict, options=tf.RunOptions.NO_TRACE)
				
	def run_value(self, states, concats=None, lstm_state=None):
		feed_dict = { 
				self.state_batch : states, 
				self._initial_lstm_state: lstm_state if lstm_state is not None else self._empty_lstm_state
			}
		if self._concat_size > 0:
			feed_dict.update( { self._concat : concats } )
		# return values, lstm_state
		return self._session.run(fetches=[self.value_batch, self._lstm_state], feed_dict=feed_dict, options=tf.RunOptions.NO_TRACE)
				
	def train(self, states, actions, rewards, values, policies, cross_entropies, discounted_cumulative_rewards, generalized_advantage_estimators, lstm_state=None, concats=None, reward_prediction_states=None, reward_prediction_target=None):
		self.train_count += len(states)
		feed_dict = self.build_feed(states, actions, rewards, values, policies, cross_entropies, discounted_cumulative_rewards, generalized_advantage_estimators, lstm_state, concats, reward_prediction_states, reward_prediction_target)
		self._session.run(fetches=self.train_op, feed_dict=feed_dict, options=tf.RunOptions.NO_TRACE) # Calculate gradients and copy them to global network
		
	def build_feed(self, states, actions, rewards, values, policies, cross_entropies, discounted_cumulative_rewards, generalized_advantage_estimators, lstm_state, concats, reward_prediction_states, reward_prediction_target):
		cross_entropies = np.reshape(cross_entropies,[-1,1])
		values = np.reshape(values,[-1,1])
		if flags.use_GAE: # Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).
			advantages = np.reshape(generalized_advantage_estimators,[-1,1])
			cumulative_rewards = advantages + values
		else:
			cumulative_rewards = np.reshape(discounted_cumulative_rewards,[-1,1])
			advantages = cumulative_rewards - values
		feed_dict={
					self.state_batch: states,
					self.action_batch: actions,
					self.cumulative_reward_batch: cumulative_rewards,
					self.advantage_batch: advantages,
					self.old_value_batch: values,
					# self.old_policy_batch: policies,
					self.old_cross_entropy_batch: cross_entropies,
					self._initial_lstm_state: lstm_state if lstm_state is not None else self._empty_lstm_state
				}
		if self._concat_size > 0:
			feed_dict.update( {self._concat : concats} )
		if self.predict_reward:
			feed_dict.update( {
				self._reward_prediction_states: reward_prediction_states,
				self.reward_prediction_labels: reward_prediction_target
			} )
		return feed_dict