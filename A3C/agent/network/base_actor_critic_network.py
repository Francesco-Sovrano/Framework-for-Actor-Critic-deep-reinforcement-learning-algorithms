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
	def __init__(self, session, id, state_shape, action_shape, entropy_beta, clip, device, predict_reward, concat_size=0, training=True, parent_id=None):
		self.train_count = 0
		self.entropy_beta = entropy_beta
		self.clip = clip
		self.predict_reward = predict_reward
		# initialize
		self.training = training
		self.session = session
		self.device = device # gpu or cpu
		self.id = id # model id
		self.parent_id = parent_id if parent_id is not None else id # used for sharing layers with other models in hierarchy, if any
		self.is_root = self.parent_id == self.id
		self.policy_size = action_shape[0] # number of actions to take
		self.policy_depth = action_shape[1] if len(action_shape) > 1 else 0 # number of discrete action types: set 0 for continuous control
		self.concat_size = concat_size # the size of the vector concatenated with the CNN output before entering the LSTM
		self.state_shape = state_shape # the shape of the input
		# lstm units
		self.lstm_units = 64 # the number of units of the LSTM
		# create the whole A3C network
		self._create_network()
	
	def is_continuous_control(self):
		return self.policy_depth <= 1
	
	def _create_network(self):
		print( "Building network {}".format(self.id) )
		# Initialize keys collections
		self.shared_keys = []
		self.update_keys = []
		# Initialize scope names
		scope_name = "Net{0}".format(self.id)
		parent_scope_name = "Net{0}".format(self.parent_id)
		# Build nets
		with tf.device(self.device):
			# [Input]
			self.state_batch = self._state_placeholder("state")
			self.concat_batch = self._concat_placeholder("concat")
			self.old_value_batch = self._value_placeholder("old_value")
			self.old_policy_batch = self._policy_placeholder("old_policy")
			self.old_action_batch = self._action_placeholder("old_action_batch")
			self.cumulative_reward_batch = self._value_placeholder("cumulative_reward")
			# [Batch Normalization]
			# _, self.state_batch_norm = self._batch_norm_layer(input=self.state_batch, scope="Global", name="State", share_trainables=False) # global
			# _, self.concat_batch_norm = self._batch_norm_layer(input=self.concat_batch, scope="Global", name="Concat{}".format("Root" if self.is_root else "Leaf"), share_trainables=False) # global
			# [Advantage]
			self.advantage_batch = tf.stop_gradient(self.cumulative_reward_batch - self.old_value_batch) # stopping gradient
			# [Layer]
			self.cnn = self._cnn_layer(input=self.state_batch, scope=parent_scope_name)
			lstm_initial_state = self._lstm_state_variables(batch_size=1, units=self.lstm_units, scope=parent_scope_name, name="Root" if self.is_root else "Leaf") # shared with family
			self.lstm, stateful_ops = self._lstm_layer(input=self.cnn, concat=self.concat_batch, initial_state=lstm_initial_state, scope=parent_scope_name, name="Root" if self.is_root else "Leaf") # shared with family
			self.backup_lstm_state_op, self.restore_lstm_state_op, self.update_lstm_state_op, self.reset_lstm_state_op = stateful_ops
			self.policy_batch = self._policy_layer(input=self.lstm, scope=scope_name)
			self.value_batch = self._value_layer(input=self.lstm, scope=parent_scope_name) # shared with family
			if self.predict_reward:
				self.reward_prediction_state_batch = self._state_placeholder("reward_prediction_state",3)
				# reusing with a different placeholder seems to cause memory leaks
				reward_prediction_cnn = self._cnn_layer(input=self.reward_prediction_state_batch, scope=scope_name)
				self.reward_prediction_logits = self._reward_prediction_layer(input=reward_prediction_cnn, scope=scope_name)
		# Sample action, after getting keys
		self.action_batch = self.sample_actions()
		# Get cnn feature mean entropy
		# self.fentropy = self.get_feature_entropy(input=self.lstm, scope=parent_scope_name, name="FEntropy_LSTM{}".format("Root" if self.is_root else "Leaf"), share_trainables=False)
		# Print shapes
		print( "    [{}]Input shape: {}".format(self.id, self.state_batch.get_shape()) )
		print( "    [{}]Concatenation shape: {}".format(self.id, self.concat_batch.get_shape()) )
		print( "    [{}]Tower shape: {}".format(self.id, self.cnn.get_shape()) )
		print( "    [{}]LSTM shape: {}".format(self.id, self.lstm.get_shape()) )
		print( "    [{}]Policy shape: {}".format(self.id, self.policy_batch.get_shape()) )
		print( "    [{}]Value shape: {}".format(self.id, self.value_batch.get_shape()) )
		if self.predict_reward:
			print( "    [{}]Reward prediction logits shape: {}".format(self.id, self.reward_prediction_logits.get_shape()) )
		print( "    [{}]Action shape: {}".format(self.id, self.action_batch.get_shape()) )

	def _separate(self, input, value):
		input_shape = tf.shape(input)
		true_labels = tf.ones(input_shape)
		false_labels = tf.zeros(input_shape)
		mask = tf.where(tf.greater_equal(input, value), true_labels, false_labels)
		greater_equal = mask*input
		lower = input - greater_equal
		return greater_equal, lower
		
	def get_feature_entropy(self, input, scope, name="", share_trainables=True): # feature entropy measures how much the input is uncommon
		with tf.device(self.device):
			batch_norm, _ = self._batch_norm_layer(input=input, scope=scope, name=name)
			feature_distribution = tf.distributions.Normal(batch_norm.moving_mean, tf.sqrt(batch_norm.moving_variance))
			return -feature_distribution.log_prob(input) # probability density function
		
	def _batch_norm_layer(self, input, scope, name="", share_trainables=True):
		with tf.variable_scope(scope), tf.variable_scope("BatchNorm{}".format(name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "    [{}]Building scope: {}".format(self.id, variable_scope.name) )
			batch_norm = tf.layers.BatchNormalization(renorm=True) # renorm because minibaches are too small
			norm_input = batch_norm.apply(input,training=self.training)
			# update keys
			if share_trainables:
				self.shared_keys += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=variable_scope.name)
			self.update_keys += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=variable_scope.name)
			# return result
			return batch_norm, norm_input
		
	# relu vs leaky_relu <https://www.reddit.com/r/MachineLearning/comments/4znzvo/what_are_the_advantages_of_relu_over_the/>
	def _cnn_layer(self, input, scope, name="", share_trainables=True):
		with tf.variable_scope(scope), tf.variable_scope("CNN{}".format(name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "    [{}]Building scope: {}".format(self.id, variable_scope.name) )
			# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=16, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.leaky_relu) # xavier initializer
			# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=32, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.leaky_relu) # xavier initializer
			input = tf.layers.conv2d( inputs=input, filters=16, kernel_size=(3,3), padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.variance_scaling )
			input = tf.layers.conv2d( inputs=input, filters=8, kernel_size=(3,3), padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.variance_scaling )
			# update keys
			if share_trainables:
				self.shared_keys += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=variable_scope.name)
			self.update_keys += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=variable_scope.name)
			# return result
			return input
	
	# https://stackoverflow.com/questions/38441589/is-rnn-initial-state-reset-for-subsequent-mini-batches
	def _lstm_state_variables(self, batch_size, units, scope, name=""):
		with tf.variable_scope(scope), tf.variable_scope("LSTM_State{}".format(name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "    [{}]Building scope: {}".format(self.id, variable_scope.name) )
			c = tf.Variable(tf.zeros([batch_size, units]), trainable=False)
			h = tf.Variable(tf.zeros([batch_size, units]), trainable=False)
			return tf.contrib.rnn.LSTMStateTuple(c,h)
	
	def _lstm_state_update_op(self, destination, source):
		# Add an operation to update the train states with the last state tensors
		c = tf.assign(destination[0],source[0])
		h = tf.assign(destination[1],source[1])
		# Return a tuple in order to combine all update_ops into a single operation.
		# The tuple's actual value should not be used.
		return (c,h)
		
	def _lstm_state_variables_copy(self, state):
		c = tf.Variable(state[0].initialized_value(), trainable=False)
		h = tf.Variable(state[1].initialized_value(), trainable=False)
		return tf.contrib.rnn.LSTMStateTuple(c,h)
		
	def _lstm_stateful_ops(self, initial_state, final_state):
		# Add an operation to backup the train states
		initial_state_backup = self._lstm_state_variables_copy(initial_state)
		backup_lstm_state_op = self._lstm_state_update_op(initial_state_backup, initial_state)
		# Add an operation to reset the train states
		reset_lstm_state_op = self._lstm_state_update_op(initial_state, self._lstm_state_variables_copy(initial_state))
		restore_lstm_state_op = self._lstm_state_update_op(initial_state, initial_state_backup)
		# Add an operation to update the train states with the last state tensors
		update_lstm_state_op = self._lstm_state_update_op(initial_state, final_state)
		# Return result
		return backup_lstm_state_op, restore_lstm_state_op, update_lstm_state_op, reset_lstm_state_op
	
	def _lstm_layer(self, input, concat, initial_state, scope, name="", share_trainables=True):
		with tf.variable_scope(scope), tf.variable_scope("LSTM_Network{}".format(name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "    [{}]Building scope: {}".format(self.id, variable_scope.name) )
			sequence_length = [tf.shape(input)[0]]
			state_shape = initial_state[0].get_shape().as_list()
			batch_size = state_shape[0]
			units = state_shape[1]
			input = tf.layers.flatten(input) # shape: (batch,w*h*depth)
			# input = tf.contrib.model_pruning.masked_fully_connected(inputs=input, num_outputs=units, activation_fn=tf.nn.leaky_relu) # xavier initializer
			input = tf.layers.dense(inputs=input, units=units, activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.variance_scaling)
			if self.concat_size > 0:
				input = tf.concat([input, concat], 1) # shape: (batch, concat_size+units)
				input = tf.reshape(input, [-1, batch_size, units+self.concat_size]) # shape: (batch, 1, concat_size+units) # time_major format
			else:
				input = tf.reshape(input, [-1, batch_size, units]) # shape: (batch, 1, units) # time_major format
			# Build LSTM cell
			# lstm_cell = tf.contrib.model_pruning.MaskedBasicLSTMCell(num_units=units, forget_bias=1.0, state_is_tuple=True, activation=None)
			lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=units, forget_bias=1.0, state_is_tuple=True) # using BasicLSTMCell instead of LSTMCell
			# Unroll the LSTM
			lstm_outputs, final_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=input, initial_state=initial_state, sequence_length=sequence_length, time_major=True)
			# Dropout: https://www.nature.com/articles/s41586-018-0102-6
			lstm_outputs = tf.layers.dropout(inputs=lstm_outputs, rate=0.5)
			lstm_outputs = tf.reshape(lstm_outputs, [-1,units]) # shape: (batch, units)
			# Update keys
			if share_trainables:
				self.shared_keys += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=variable_scope.name)
			self.update_keys += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=variable_scope.name)
			# Return result
			return lstm_outputs, self._lstm_stateful_ops(initial_state, final_state)

	def _value_layer(self, input, scope, name="", share_trainables=True):
		with tf.variable_scope(scope), tf.variable_scope("Value{}".format(name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "    [{}]Building scope: {}".format(self.id, variable_scope.name) )
			input = tf.layers.dense(inputs=input, units=1, activation=None, kernel_initializer=tf.initializers.variance_scaling)
			input = tf.reshape(input,[-1]) # flatten
			# update keys
			if share_trainables:
				self.shared_keys += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=variable_scope.name)
			self.update_keys += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=variable_scope.name)
			# return result
			return input
			
	def _policy_layer(self, input, scope, name="", share_trainables=True):
		with tf.variable_scope(scope), tf.variable_scope("Policy{}".format(name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "    [{}]Building scope: {}".format(self.id, variable_scope.name) )
			if self.is_continuous_control():
				# build mean
				mu = tf.layers.dense(inputs=input, units=self.policy_size, activation=None, kernel_initializer=tf.initializers.variance_scaling) # in (-inf,inf)
				# build standard deviation
				sigma = tf.layers.dense(inputs=input, units=self.policy_size, activation=None, kernel_initializer=tf.initializers.variance_scaling) # in (-inf,inf)
				# clip mu and sigma to avoid numerical instabilities
				clipped_mu = tf.clip_by_value(mu, -1,1) # in [-1,1]
				clipped_sigma = tf.clip_by_value(tf.abs(sigma), 1e-4,1) # in [1e-4,1] # sigma must be greater than 0
				# build policy batch
				policy_batch = tf.stack([clipped_mu, clipped_sigma])
				policy_batch = tf.transpose(policy_batch, [1, 0, 2])
			else: # discrete control
				policy_batch = []
				for _ in range(self.policy_size):
					policy_batch.append(tf.layers.dense(inputs=input, units=self.policy_depth, activation=None, kernel_initializer=tf.initializers.variance_scaling))
				shape = [-1,self.policy_size,self.policy_depth] if self.policy_size > 1 else [-1,self.policy_depth]
				policy_batch = tf.reshape(policy_batch, shape)
			# update keys
			if share_trainables:
				self.shared_keys += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=variable_scope.name)
			self.update_keys += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=variable_scope.name)
			# return result
			return policy_batch
			
	def _reward_prediction_layer(self, input, scope, name="", share_trainables=True):
		with tf.variable_scope(scope), tf.variable_scope("RewardPrediction", reuse=tf.AUTO_REUSE) as variable_scope:
			print( "    [{}]Building scope: {}".format(self.id, variable_scope.name) )
			# input = tf.contrib.layers.maxout(inputs=input, num_units=1, axis=0)
			input = tf.reshape(input,[1,-1])
			input = tf.layers.dense(inputs=input, units=3, activation=None, kernel_initializer=tf.initializers.variance_scaling)
			# update keys
			if share_trainables:
				self.shared_keys += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=variable_scope.name)
			self.update_keys += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=variable_scope.name)
			# return result
			return input

	def _categorical_cross_entropy(self, samples, logits):
		return tf.nn.softmax_cross_entropy_with_logits_v2(labels=samples, logits=logits)

	def _categorical_entropy(self, logits):
		a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
		ea0 = tf.exp(a0)
		z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
		p0 = ea0 / z0
		return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)
			
	def _categorical_sample(self, logits):
		logits_shape = logits.get_shape()
		# print("    [{}]Logits shape: {}".format(self.id, logits_shape))
		u = tf.random_uniform(tf.shape(logits))
		samples = tf.argmax(logits - tf.log(-tf.log(u)), axis=-1)
		# print("    [{}]Samples shape: {}".format(self.id, samples.get_shape()))
		depth = logits_shape.as_list()[-1]
		one_hot_actions = tf.one_hot(samples, depth)
		one_hot_actions.set_shape(logits_shape)
		return one_hot_actions
			
	def _reward_prediction_loss(self):
		self.reward_prediction_labels = self._reward_prediction_target_placeholder("reward_prediction_target",1)
		return tf.reduce_sum(self._categorical_cross_entropy(samples=self.reward_prediction_labels, logits=self.reward_prediction_logits))
		
	def sample_actions(self):
		with tf.device(self.device):
			if self.is_continuous_control():
				new_policy_batch = tf.transpose(self.policy_batch, [1, 0, 2])
				new_policy_distributions = tf.distributions.Normal(new_policy_batch[0], new_policy_batch[1], validate_args=False) # validate_args is computationally expensive
				action_batch = tf.clip_by_value(new_policy_distributions.sample(), -1,1) # Sample action batch in forward direction, use old action in backward direction
			else: # discrete control
				action_batch = self._categorical_sample(self.policy_batch) # Sample action batch in forward direction, use old action in backward direction
			return action_batch
		
	def prepare_loss(self):
		with tf.device(self.device):
			print( "Preparing loss {}".format(self.id) )
			# [Entropy]
			if self.is_continuous_control():
				# Old policy
				old_policy_batch = tf.transpose(self.old_policy_batch, [1, 0, 2])
				old_policy_distributions = tf.distributions.Normal(old_policy_batch[0], old_policy_batch[1], validate_args=False) # validate_args is computationally expensive
				old_cross_entropy_batch = -old_policy_distributions.log_prob(self.old_action_batch) # probability density function
				# New policy
				new_policy_batch = tf.transpose(self.policy_batch, [1, 0, 2])
				new_policy_distributions = tf.distributions.Normal(new_policy_batch[0], new_policy_batch[1], validate_args=False) # validate_args is computationally expensive
				new_cross_entropy_batch = -new_policy_distributions.log_prob(self.old_action_batch) # probability density function
				new_entropy_batch = new_policy_distributions.entropy()
			else: # discrete control
				# Old policy
				old_cross_entropy_batch = self._categorical_cross_entropy(samples=self.old_action_batch, logits=self.old_policy_batch)
				# New policy
				new_cross_entropy_batch = self._categorical_cross_entropy(samples=self.old_action_batch, logits=self.policy_batch)
				new_entropy_batch = self._categorical_entropy(self.policy_batch)
			# new_entropy_batch = self.fentropy
			# [Actor loss]
			policy_loss = PolicyLoss(cliprange=self.clip, cross_entropy=new_cross_entropy_batch, old_cross_entropy=old_cross_entropy_batch, advantage=self.advantage_batch, entropy=new_entropy_batch, entropy_beta=self.entropy_beta).get()
			# [Critic loss]
			value_loss = ValueLoss(cliprange=self.clip, value=self.value_batch, old_value=self.old_value_batch, reward=self.cumulative_reward_batch).get()
			value_loss *= flags.value_coefficient # usually critic has lower learning rate
			# [Extra loss]
			extra_loss = tf.constant(0.)
			if self.predict_reward:
				extra_loss += self._reward_prediction_loss()
			return (policy_loss, value_loss, extra_loss)
			
	def bind_sync(self, src_network, name=None):
		with tf.device(self.device), tf.name_scope(name, "Sync{0}".format(self.id),[]) as name:
			src_vars = src_network.get_shared_keys()
			dst_vars = self.get_shared_keys()
			sync_ops = []
			for(src_var, dst_var) in zip(src_vars, dst_vars):
				sync_op = tf.assign(dst_var, src_var)
				sync_ops.append(sync_op)
			return tf.group(*sync_ops, name=name)
				
	def sync(self, sync):
		self.session.run(fetches=sync)

	def minimize_local_loss(self, optimizer, global_step, global_var_list): # minimize loss and apply gradients to global vars.
		self.policy_loss, self.value_loss, self.extra_loss = self.prepare_loss()
		with tf.device(self.device) and tf.control_dependencies(self.update_keys): # control_dependencies is for batch normalization
			var_refs = [v._ref() for v in self.get_shared_keys()]
			total_loss = self.policy_loss + self.value_loss + self.extra_loss
			local_gradients = tf.gradients(total_loss, var_refs, gate_gradients=False, aggregation_method=None, colocate_gradients_with_ops=False)
			if flags.grad_norm_clip > 0:
				local_gradients, _ = tf.clip_by_global_norm(local_gradients, flags.grad_norm_clip)
			grads_and_vars = list(zip(local_gradients, global_var_list))
			self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

	def get_shared_keys(self):
		return self.shared_keys # get model variables
		
	def reset(self):
		self.session.run(fetches=self.reset_lstm_state_op)
		self.session.run(fetches=self.backup_lstm_state_op) # Backup new initial state after reset, in a different run in order to preserve execution order!
		
	def predict_action(self, states, concats=None):
		feed_dict = { self.state_batch : states }
		if self.concat_size > 0:
			feed_dict.update( { self.concat_batch : concats } )
		action_batch, value_batch, policy_batch = self.session.run(fetches=[self.action_batch, self.value_batch, self.policy_batch], feed_dict=feed_dict)
		self.session.run(fetches=self.update_lstm_state_op) # Update LSTM state, in a different run in order to preserve execution order!
		return action_batch, value_batch, policy_batch
				
	def predict_value(self, states, concats=None):
		feed_dict = { self.state_batch : states }
		if self.concat_size > 0:
			feed_dict.update( { self.concat_batch : concats } )
		#return value_batch
		return self.session.run(fetches=[self.value_batch], feed_dict=feed_dict)
				
	def train(self, replay, states, actions, rewards, values, policies, discounted_cumulative_rewards, generalized_advantage_estimators, concats=None, reward_prediction_states=None, reward_prediction_target=None):
		self.train_count += len(states)
		feed_dict = self.build_train_feed(states, actions, rewards, values, policies, discounted_cumulative_rewards, generalized_advantage_estimators, concats, reward_prediction_states, reward_prediction_target)
		if replay:
			self.session.run(fetches=self.reset_lstm_state_op) # Reset initial state, in a different run in order to preserve execution order!
		else:
			self.session.run(fetches=self.restore_lstm_state_op) # Restore old initial state backup, in a different run in order to preserve execution order!
		_, policy_loss, value_loss, extra_loss = self.session.run(fetches=[self.train_op, self.policy_loss, self.value_loss, self.extra_loss], feed_dict=feed_dict) # Minimize gradients and copy them to global network
		if replay:
			self.session.run(fetches=self.restore_lstm_state_op) # Restore old initial state backup, in a different run in order to preserve execution order!
		else:
			self.session.run(fetches=self.update_lstm_state_op) # Update LSTM state, in a different run in order to preserve execution order!
			self.session.run(fetches=self.backup_lstm_state_op) # Backup new initial state after updating it, in a different run in order to preserve execution order!
		if self.predict_reward:
			return {"actor": policy_loss, "critic": value_loss, "extra": extra_loss}
		else:
			return {"actor": policy_loss, "critic": value_loss}
		
	def build_train_feed(self, states, actions, rewards, values, policies, discounted_cumulative_rewards, generalized_advantage_estimators, concats, reward_prediction_states, reward_prediction_target):
		if flags.use_GAE: # Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).
			advantages = np.reshape(generalized_advantage_estimators,[-1])
			values = np.reshape(values,[-1])
			cumulative_rewards = advantages + values
		else:
			cumulative_rewards = np.reshape(discounted_cumulative_rewards,[-1])
		feed_dict={
				self.state_batch: states,
				self.cumulative_reward_batch: cumulative_rewards,
				self.old_value_batch: values,
				self.old_policy_batch: policies,
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
		
	def _reward_prediction_target_placeholder(self, name=None, batch_size=None):
		return tf.placeholder(dtype=tf.float32, shape=[batch_size,3], name=name)
		
	def _policy_placeholder(self, name=None, batch_size=None):
		if self.is_continuous_control():
			shape = [batch_size,2,self.policy_size]
		else: # discrete control
			shape = [batch_size,self.policy_size,self.policy_depth] if self.policy_size > 1 else [batch_size,self.policy_depth]
		return tf.placeholder(dtype=tf.float32, shape=shape, name=name)
			
	def _action_placeholder(self, name=None, batch_size=None):
		shape = [batch_size]
		if self.policy_size > 1:
			shape.append(self.policy_size)
		if self.policy_depth > 1:
			shape.append(self.policy_depth)
		return tf.placeholder(dtype=tf.float32, shape=shape, name=name)
		
	def _value_placeholder(self, name=None, batch_size=None):
		return tf.placeholder(dtype=tf.float32, shape=[batch_size], name=name)
		
	def _concat_placeholder(self, name=None, batch_size=None):
		input=[tf.zeros(self.concat_size)] # default value
		shape=[batch_size,self.concat_size]
		return tf.placeholder_with_default(input=input, shape=shape, name=name) # with default we can use batch normalization directly on it

	def _state_placeholder(self, name=None, batch_size=None):
		input=[tf.zeros(self.state_shape)] # default value
		shape=np.concatenate([[batch_size], self.state_shape], 0)
		return tf.placeholder_with_default(input=input, shape=shape, name=name) # with default we can use batch normalization directly on it