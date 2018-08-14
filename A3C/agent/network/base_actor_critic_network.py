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
		self.concat_size = concat_size # the size of the vector concatenated with the CNN output before entering the LSTM
		self._state_shape = state_shape # the shape of the input
		# lstm units
		self._lstm_units = 64 # the number of units of the LSTM
		# create the whole A3C network
		self._create_network()
	
	def _create_network(self):
		print( "Building network {}".format(self._id) )
		# Batch placeholders
		self.state_batch = self._state_placeholder("state")
		self.concat_batch = self._concat_placeholder("concat")
		self.old_value_batch = self._value_placeholder("old_value")
		self.old_cross_entropy_batch = self._entropy_placeholder("old_cross_entropy")
		self.cumulative_reward_batch = self._value_placeholder("cumulative_reward")
		self.old_action_batch = self._action_placeholder("old_action_batch")
		# self.episode_reward = self._singleton_placeholder("episode_reward")
		# self.min_value = tf.Variable(float("+inf"), trainable=False)
		# self.max_value = tf.Variable(float("-inf"), trainable=False)
		# self.min_op = tf.assign(self.min_value, tf.minimum(self.min_value, self.episode_reward))
		# self.max_op = tf.assign(self.max_value, tf.maximum(self.max_value, self.episode_reward))
		print( "    [{}]Input shape: {}".format(self._id, self.state_batch.get_shape()) )
		print( "    [{}]Concatenation shape: {}".format(self._id, self.concat_batch.get_shape()) )
		# Build nets
		scope_name = "net_{0}".format(self._id)
		with tf.device(self._device), tf.variable_scope(scope_name) as scope:
			self._build_base()
			if self.predict_reward:
				self.reward_prediction_cross_entropy = self._build_reward_prediction()
		# Get keys
		self.train_keys = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
		# self.update_keys = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope_name) # for batch normalization
		
	def _build_base(self):
		# [CNN tower]
		tower = self._convolutive_layers(self.state_batch)
		print( "    [{}]Tower shape: {}".format(self._id, tower.get_shape()) )
		# [LSTM]
		lstm, self._lstm_state = self._lstm_layers(tower, self.concat_batch)
		print( "    [{}]LSTM shape: {}".format(self._id, lstm.get_shape()) )
		# [Policy]
		self.action_batch, self.cross_entropy_batch, self.policy_distributions = self._policy_layers(lstm)
		print( "    [{}]Action shape: {}".format(self._id, self.action_batch.get_shape()) )
		print( "    [{}]Cross Entropy shape: {}".format(self._id, self.cross_entropy_batch.get_shape()) )
		# [Value]
		self.value_batch = self._value_layers(lstm)
		print( "    [{}]Value shape: {}".format(self._id, self.value_batch.get_shape()) )
		
	def _build_reward_prediction(self):
		# Memory leak fix: https://github.com/tensorflow/tensorflow/issues/12704
		self.reward_prediction_labels = self._reward_prediction_target_placeholder("reward_prediction_target",1)
		self.reward_prediction_state_batch = self._state_placeholder("reward_prediction_state",3)
		input = self._convolutive_layers(self.reward_prediction_state_batch) # do it outside reward_prediction variable scope
		with tf.variable_scope("reward_prediction{0}".format(self._id), reuse=tf.AUTO_REUSE) as scope:
			# input = tf.contrib.layers.maxout(inputs=input, num_units=1, axis=0)
			input = tf.reshape(input,[1,-1])
			logits = tf.layers.dense(inputs=input, units=3, activation=None, kernel_initializer=tf.initializers.variance_scaling)
			print( "    [{}]Reward prediction logits shape: {}".format(self._id, logits.get_shape()) )
			# policies = tf.contrib.layers.softmax(logits)
			return self.get_cross_entropy(samples=self.reward_prediction_labels, distributions=logits)
		
	def _convolutive_layers(self, input):
		with tf.variable_scope("base_conv{0}".format(self._id), reuse=tf.AUTO_REUSE) as scope:
			# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=16, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.relu) # xavier initializer
			# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=32, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.relu) # xavier initializer
			input = tf.layers.conv2d( inputs=input, filters=16, kernel_size=(3,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )
			input = tf.layers.conv2d( inputs=input, filters=8, kernel_size=(3,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )
			# input = tf.layers.batch_normalization(inputs=input, renorm=True, training=self._training) # renorm because minibaches are too small
			return input
	
	def _lstm_layers(self, input, concat):
		with tf.variable_scope("base_lstm{0}".format(self._id), reuse=tf.AUTO_REUSE) as scope:
			input = tf.layers.flatten(input) # shape: (batch,w*h*depth)
			# input = tf.contrib.model_pruning.masked_fully_connected(inputs=input, num_outputs=self._lstm_units, activation_fn=tf.nn.relu) # xavier initializer
			input = tf.layers.dense(inputs=input, units=self._lstm_units, activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling)
			step_size = tf.shape(input)[:1] # shape: (batch)
			if self.concat_size > 0:
				input = tf.concat([input, concat], 1) # shape: (batch, concat_size+lstm_units)
				input = tf.reshape(input, [1, -1, self._lstm_units+self.concat_size]) # shape: (1, batch, concat_size+lstm_units)
			else:
				input = tf.reshape(input, [1, -1, self._lstm_units]) # shape: (1, batch, lstm_units)

			# self._lstm_cell = tf.contrib.model_pruning.MaskedBasicLSTMCell(num_units=self._lstm_units, forget_bias=1.0, state_is_tuple=True, activation=None)
			self._lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self._lstm_units)
			self._initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self._lstm_state_placeholder("lstm_tuple_1",1), self._lstm_state_placeholder("lstm_tuple_2",1))
			self._empty_lstm_state = (np.zeros([1,self._lstm_units], dtype=np.float32),np.zeros([1,self._lstm_units], dtype=np.float32))
			lstm_outputs, lstm_state = tf.nn.dynamic_rnn(cell=self._lstm_cell, inputs=input, initial_state=self._initial_lstm_state, sequence_length=step_size, time_major = False, scope = scope)
			# Dropout: https://www.nature.com/articles/s41586-018-0102-6
			lstm_outputs = tf.layers.dropout(inputs=lstm_outputs, rate=0.5)			
			lstm_outputs = tf.reshape(lstm_outputs, [-1,self._lstm_units]) # shape: (batch, lstm_units)
			return lstm_outputs, lstm_state

	def _value_layers(self, input): # Value (output)
		with tf.variable_scope("base_value{0}".format(self._id), reuse=tf.AUTO_REUSE) as scope:
			input = tf.layers.dense(inputs=input, units=1, activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling)
			# input = tf.clip_by_value(input, 0,1)
			# min = tf.where(tf.equal(self.min_value,float("+inf")), 0., self.min_value)
			# max = tf.where(tf.equal(self.max_value,float("-inf")), 0., self.max_value)
			# input = min + input*(max-min)
			return tf.reshape(input,[-1]) # flatten
			
	def _policy_layers(self, input): # Policy (output)
		with tf.variable_scope("base_policy{0}".format(self._id), reuse=tf.AUTO_REUSE) as scope:
			if self.policy_depth < 1: # continuous control
				# mean
				mu = tf.layers.dense(inputs=input, units=self.policy_length, activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling)
				# standard deviation
				sigma = tf.layers.dense(inputs=input, units=self.policy_length, activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling)
				# clip mu and sigma in order to avoid numeric instabilities
				clipped_mu = tf.clip_by_value(mu, 0,1)
				clipped_sigma = tf.clip_by_value(sigma, 1e-4,np.sqrt(0.5)) # sigma MUST be greater than 0
				# build distribution
				policy_distributions = tf.distributions.Normal(clipped_mu, clipped_sigma, validate_args=False) # validate_args is computationally expensive
				# sample action batch in forward direction, use old action in backward direction
				action_batch = tf.clip_by_value(policy_distributions.sample(),0,1)
				# compute entropies
				cross_entropy_batch = -policy_distributions.log_prob(action_batch) # probability density function
			else: # discrete control
				logits = tf.layers.dense(inputs=input, units=self.policy_length, activation=None, kernel_initializer=tf.initializers.variance_scaling)
				# policy_distributions = tf.contrib.layers.softmax(logits)
				policy_distributions = logits
				# sample action batch in forward direction, use old action in backward direction
				action_batch = self.sample_one_hot_actions(logits)
				# softmax_cross_entropy_with_logits_v2 stops labels gradient
				cross_entropy_batch = self.get_cross_entropy(samples=action_batch, distributions=policy_distributions)
			return action_batch, cross_entropy_batch, policy_distributions

	def get_cross_entropy(self, samples, distributions):
		return tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.identity(samples), logits=distributions) # deep-copy labels using tf.identity because tf.nn.softmax_cross_entropy_with_logits_v2 stops labels gradient even if labels == logits
		# return -tf.reduce_sum(samples*tf.log(tf.clip_by_value(distributions, 1e-20, 1.0)), axis=-1) # Avoid NaN with clipping when value in pi becomes zero
			
	def sample_one_hot_actions(self, logits):
		print( "    [{}]Logits shape: {}".format(self._id, logits.get_shape()) )
		samples = tf.multinomial(logits, 1)
		print( "    [{}]Samples shape: {}".format(self._id, samples.get_shape()) )
		depth = logits.get_shape().as_list()[-1]
		one_hot_actions = tf.one_hot(samples, depth)
		return tf.layers.flatten(one_hot_actions)
		
	def prepare_loss(self):
		# Get new entropy by using old actions and new policy_distribution
		if self.policy_depth < 1: # continuous control
			cross_entropy_batch = -self.policy_distributions.log_prob(self.old_action_batch) # probability density function
			entropy_batch = self.policy_distributions.entropy()
		else: # discrete control
			policy_batch = tf.contrib.layers.softmax(self.policy_distributions)
			cross_entropy_batch = self.get_cross_entropy(samples=self.old_action_batch, distributions=self.policy_distributions)
			entropy_batch = self.get_cross_entropy(samples=policy_batch, distributions=self.policy_distributions)
		with tf.device(self._device):
			# Build losses
			self.policy_loss = PolicyLoss(
				cliprange=self.clip,
				cross_entropy=cross_entropy_batch,
				old_cross_entropy=self.old_cross_entropy_batch,
				advantage=self.cumulative_reward_batch - self.old_value_batch,
				entropy=entropy_batch,
				entropy_beta=self.entropy_beta
			)
			self.value_loss = ValueLoss(
				cliprange=self.clip, 
				value=self.value_batch, 
				old_value=self.old_value_batch, 
				reward=self.cumulative_reward_batch
			)
			# Compute total loss
			self.policy_loss = self.policy_loss.get()
			self.value_loss = flags.value_coefficient*self.value_loss.get()
			self.total_loss = self.policy_loss + self.value_loss
			if self.predict_reward:
				self.total_loss += tf.reduce_sum(self.reward_prediction_cross_entropy)
			
	def bind_sync(self, src_network):
		src_vars = src_network.get_vars()
		dst_vars = self.get_vars()
		sync_ops = []
		with tf.device(self._device):
			# update trainable variables
			for(src_var, dst_var) in zip(src_vars, dst_vars):
				sync_op = tf.assign(ref=dst_var, value=src_var, use_locking=False) # no need for locking
				sync_ops.append(sync_op)
			return tf.group(*sync_ops)

	def sync(self, sync):
		self._session.run(fetches=sync)

	def minimize_local(self, optimizer, global_step, global_var_list):
		"""
		minimize loss and apply gradients to global vars.
		"""
		# with tf.device(self._device) and tf.control_dependencies(self.update_keys): # control_dependencies is for batch normalization
		with tf.device(self._device):
			loss = self.total_loss
			local_var_list = self.get_vars()
			var_refs = [v._ref() for v in local_var_list]
			local_gradients = tf.gradients(loss, var_refs, gate_gradients=False, aggregation_method=None, colocate_gradients_with_ops=False)
			if flags.grad_norm_clip > 0:
				local_gradients, _ = tf.clip_by_global_norm(local_gradients, flags.grad_norm_clip)
			grads_and_vars = list(zip(local_gradients, global_var_list))
			self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

	def get_vars(self):
		return self.train_keys # get model variables
		
	def run_action_and_value(self, states, concats=None, lstm_state=None):
		feed_dict = { 
				self.state_batch : states,
				self._initial_lstm_state: lstm_state if lstm_state is not None else self._empty_lstm_state
			}
		if self.concat_size > 0:
			feed_dict.update( { self.concat_batch : concats } )
		# return action_batch, value_batch, entropy_batch, cross_entropy_batch, lstm_state
		return self._session.run(fetches=[self.action_batch, self.value_batch, self.cross_entropy_batch, self._lstm_state], feed_dict=feed_dict)
				
	def run_value(self, states, concats=None, lstm_state=None):
		feed_dict = { 
				self.state_batch : states, 
				self._initial_lstm_state: lstm_state if lstm_state is not None else self._empty_lstm_state
			}
		if self.concat_size > 0:
			feed_dict.update( { self.concat_batch : concats } )
		#return value_batch, lstm_state
		return self._session.run(fetches=[self.value_batch, self._lstm_state], feed_dict=feed_dict)
				
	def train(self, states, actions, rewards, values, cross_entropies, discounted_cumulative_rewards, generalized_advantage_estimators, lstm_state=None, concats=None, reward_prediction_states=None, reward_prediction_target=None):
		self.train_count += len(states)
		feed_dict = self.build_feed(states, actions, rewards, values, cross_entropies, discounted_cumulative_rewards, generalized_advantage_estimators, lstm_state, concats, reward_prediction_states, reward_prediction_target)
		_, total_loss, policy_loss, value_loss = self._session.run(fetches=[self.train_op,self.total_loss,self.policy_loss,self.value_loss], feed_dict=feed_dict) # Calculate gradients and copy them to global network
		return total_loss, policy_loss, value_loss
		
	def build_feed(self, states, actions, rewards, values, cross_entropies, discounted_cumulative_rewards, generalized_advantage_estimators, lstm_state, concats, reward_prediction_states, reward_prediction_target):
		cross_entropies = np.reshape(cross_entropies,[-1,self.policy_length if self.policy_depth == 0 else self.policy_depth])
		values = np.reshape(values,[-1,1])
		if flags.use_GAE: # Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).
			advantages = np.reshape(generalized_advantage_estimators,[-1,1])
			cumulative_rewards = advantages + values
		else:
			cumulative_rewards = np.reshape(discounted_cumulative_rewards,[-1,1])
			# advantages = cumulative_rewards - values
		feed_dict={
				self.state_batch: states,
				self.cumulative_reward_batch: cumulative_rewards,
				self.old_value_batch: values,
				self.old_cross_entropy_batch: cross_entropies,
				self._initial_lstm_state: lstm_state if lstm_state is not None else self._empty_lstm_state,
				self.old_action_batch: actions
			}
		if self.concat_size > 0:
			feed_dict.update( {self.concat_batch : concats} )
		if self.predict_reward:
			feed_dict.update( {
				self.reward_prediction_state_batch: reward_prediction_states,
				self.reward_prediction_labels: reward_prediction_target
			} )
		return feed_dict
		
	def _lstm_state_placeholder(self, name=None, batch_size=None):
		return tf.placeholder(dtype=tf.float32, shape=[batch_size, self._lstm_units], name=name)
		
	def _reward_prediction_target_placeholder(self, name=None, batch_size=None):
		return tf.placeholder(dtype=tf.float32, shape=[batch_size,3], name=name)
		
	def _state_placeholder(self, name=None, batch_size=None):
		return tf.placeholder(dtype=tf.float32, shape=np.concatenate([[batch_size], self._state_shape], 0), name=name)
		
	def _entropy_placeholder(self, name=None, batch_size=None):
		if self.policy_depth == 0:
			return tf.placeholder(dtype=tf.float32, shape=[batch_size,self.policy_length], name=name)
		else:
			return tf.placeholder(dtype=tf.float32, shape=[batch_size,self.policy_depth], name=name)
			
	def _action_placeholder(self, name=None, batch_size=None):
		return tf.placeholder(dtype=tf.float32, shape=[batch_size,self.policy_length], name=name)
		
	def _value_placeholder(self, name=None, batch_size=None):
		return tf.placeholder(dtype=tf.float32, shape=[batch_size,1], name=name)
		
	def _concat_placeholder(self, name=None, batch_size=None):
		return tf.placeholder(dtype=tf.float32, shape=[batch_size,self.concat_size], name=name)
