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
from utils.distributions import Categorical, Normal

class BaseAC_Network(object):
	lstm_units = 64 # the number of units of the LSTM
		
	def __init__(self, session, id, state_shape, action_shape, clip, device, predict_reward, concat_size=0, beta=None, training=True, parent=None, sibling=None):
		self.train_count = 0
		self.beta = beta if beta is not None else flags.beta
		self.clip = clip
		self.predict_reward = predict_reward
		# initialize
		self.training = training
		self.session = session
		self.device = device # gpu or cpu
		self.id = id # model id
		self.parent = parent if parent is not None else self # used for sharing with other models in hierarchy, if any
		self.sibling = sibling if sibling is not None else self # used for sharing with other models in hierarchy, if any
		self.policy_size = action_shape[0] # number of actions to take
		self.policy_depth = action_shape[1] if len(action_shape) > 1 else 0 # number of discrete action types: set 0 for continuous control
		self.concat_size = concat_size # the size of the vector concatenated with the CNN output before entering the LSTM
		self.state_shape = state_shape # the shape of the input
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
		parent_scope_name = "Net{0}".format(self.parent.id)
		sibling_scope_name = "Net{0}".format(self.sibling.id)
		# Build nets
		with tf.device(self.device):
			# [Input]
			self.state_batch = self._state_placeholder("state")
			self.concat_batch = self._concat_placeholder("concat")
			self.old_value_batch = self._value_placeholder("old_value")
			self.old_policy_batch = self._policy_placeholder("old_policy")
			self.old_action_batch = self._action_placeholder("old_action_batch")
			self.cumulative_reward_batch = self._value_placeholder("cumulative_reward")
			self.advantage_batch = self._value_placeholder("advantage")
			self.lstm_initial_state = self._lstm_state_placeholder(batch_size=1, units=self.lstm_units, name="initial_lstm_state") # for stateful lstm
			self.lstm_default_state = self._lstm_default_state(batch_size=1, units=self.lstm_units)
			# [Batch Normalization]
			# _, self.state_batch_norm = self._batch_norm_layer(input=self.state_batch, scope="Global", name="State", share_trainables=False) # global
			# [CNN]
			self.cnn = self._cnn_layer(input=self.state_batch, scope=scope_name)
			# [Concat]
			self.concat = self._concat_layer(input=self.cnn, concat=self.concat_batch, units=self.lstm_units, scope=scope_name)
			# [LSTM]
			self.lstm, self.lstm_final_state = self._lstm_layer(input=self.concat, initial_state=self.lstm_initial_state, scope=scope_name)
			# [Policy]
			self.policy_batch = self._policy_layer(input=self.lstm, scope=scope_name)
			# [Value]
			self.value_batch = self._value_layer(input=self.lstm, scope=scope_name)
			# [Reward Prediction]
			if self.predict_reward:
				self.reward_prediction_state_batch = self._state_placeholder("reward_prediction_state")
				# reusing with a different placeholder seems to cause memory leaks
				reward_prediction_cnn = self._cnn_layer(input=self.reward_prediction_state_batch, scope=scope_name)
				self.reward_prediction_logits = self._reward_prediction_layer(input=reward_prediction_cnn, scope=scope_name)
		# Sample action, after getting keys
		self.action_batch = self.sample_actions()
		# Get cnn feature mean entropy
		# self.fentropy = self.get_feature_entropy(input=self.lstm, scope=scope_name)
		# Print shapes
		print( "    [{}]Input shape: {}".format(self.id, self.state_batch.get_shape()) )
		print( "    [{}]Concatenation shape: {}".format(self.id, self.concat_batch.get_shape()) )
		print( "    [{}]Tower shape: {}".format(self.id, self.cnn.get_shape()) )
		print( "    [{}]Concat shape: {}".format(self.id, self.concat.get_shape()) )
		print( "    [{}]LSTM shape: {}".format(self.id, self.lstm.get_shape()) )
		print( "    [{}]Policy shape: {}".format(self.id, self.policy_batch.get_shape()) )
		print( "    [{}]Value shape: {}".format(self.id, self.value_batch.get_shape()) )
		if self.predict_reward:
			print( "    [{}]Reward prediction logits shape: {}".format(self.id, self.reward_prediction_logits.get_shape()) )
		print( "    [{}]Action shape: {}".format(self.id, self.action_batch.get_shape()) )
		# Prepare loss
		if self.training:
			self.prepare_loss()
		# Give self esplicative names to outputs for easily retrieving them in frozen graph
		tf.identity(self.action_batch, name="action")
		tf.identity(self.value_batch, name="value")
		
	def get_feature_entropy(self, input, scope, name=""): # feature entropy measures how much the input is uncommon
		with tf.device(self.device):
			batch_norm, _ = self._batch_norm_layer(input=input, scope=scope, name=name, share_trainables=False)
			fentropy = Normal(batch_norm.moving_mean, tf.sqrt(batch_norm.moving_variance)).cross_entropy(input)
			fentropy = tf.layers.flatten(fentropy)
			if len(fentropy.get_shape()) > 1:
				fentropy = tf.reduce_mean(fentropy, axis=-1)
			return fentropy
			
	def _update_keys(self, scope_name, share_trainables):
		if share_trainables:
			self.shared_keys += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
		self.update_keys += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope_name)
		
	def _batch_norm_layer(self, input, scope, name="", share_trainables=True):
		with tf.variable_scope(scope), tf.variable_scope("BatchNorm{}".format(name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "    [{}]Building scope: {}".format(self.id, variable_scope.name) )
			batch_norm = tf.layers.BatchNormalization(renorm=True) # renorm because minibaches are too small
			norm_input = batch_norm.apply(input,training=self.training)
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return batch_norm, norm_input
		
	# relu vs leaky_relu <https://www.reddit.com/r/MachineLearning/comments/4znzvo/what_are_the_advantages_of_relu_over_the/>
	def _cnn_layer(self, input, scope, name="", share_trainables=True):
		with tf.variable_scope(scope), tf.variable_scope("CNN{}".format(name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "    [{}]Building scope: {}".format(self.id, variable_scope.name) )
			# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=16, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.relu) # xavier initializer
			# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=32, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.relu) # xavier initializer
			input = tf.layers.conv2d( inputs=input, filters=16, kernel_size=(3,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )
			input = tf.layers.conv2d( inputs=input, filters=8, kernel_size=(3,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return input
	
	def _concat_layer(self, input, concat, units, scope, name="", share_trainables=True):
		with tf.variable_scope(scope), tf.variable_scope("Concat{}".format(name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "    [{}]Building scope: {}".format(self.id, variable_scope.name) )
			input = tf.layers.flatten(input)
			input = tf.layers.dense(inputs=input, units=units, activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling)
			if concat.get_shape()[-1] > 0:
				concat = tf.layers.flatten(concat)
				input = tf.concat([input, concat], -1) # shape: (batch, concat_size+units)
			# Update keys
			self._update_keys(variable_scope.name, share_trainables)
			# Return result
			return input
	
	def _lstm_layer(self, input, initial_state, scope, name="", share_trainables=True):
		with tf.variable_scope(scope), tf.variable_scope("LSTM_Network{}".format(name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "    [{}]Building scope: {}".format(self.id, variable_scope.name) )
			if len(input.get_shape()) > 2:
				input = tf.layers.flatten(input)
			sequence_length = [tf.shape(input)[0]]
			state_shape = initial_state[0].get_shape().as_list()
			batch_size = state_shape[0]
			units = state_shape[1]
			# Add batch dimension
			input = tf.reshape(input, [-1, batch_size, input.get_shape().as_list()[-1]])
			# Build LSTM cell
			# lstm_cell = tf.contrib.model_pruning.MaskedBasicLSTMCell(num_units=units, forget_bias=1.0, state_is_tuple=True, activation=None)
			lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=units, state_is_tuple=True) # using BasicLSTMCell instead of LSTMCell
			# Unroll the LSTM
			lstm_state_tuple = tf.nn.rnn_cell.LSTMStateTuple(initial_state[0],initial_state[1])
			lstm_outputs, final_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=input, initial_state=lstm_state_tuple, sequence_length=sequence_length, time_major=True)
			# Dropout: https://www.nature.com/articles/s41586-018-0102-6
			lstm_outputs = tf.layers.dropout(inputs=lstm_outputs, rate=0.5)
			lstm_outputs = tf.reshape(lstm_outputs, [-1,units]) # shape: (batch, units)
			# Update keys
			self._update_keys(variable_scope.name, share_trainables)
			# Return result
			return lstm_outputs, final_state
			
	def _value_layer(self, input, scope, name="", share_trainables=True):
		with tf.variable_scope(scope), tf.variable_scope("Value{}".format(name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "    [{}]Building scope: {}".format(self.id, variable_scope.name) )
			input = tf.layers.dense(inputs=input, units=1, activation=None, kernel_initializer=tf.initializers.variance_scaling)
			input = tf.reshape(input,[-1]) # flatten
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
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
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return policy_batch
			
	def _reward_prediction_layer(self, input, scope, name="", share_trainables=True):
		with tf.variable_scope(scope), tf.variable_scope("RewardPrediction", reuse=tf.AUTO_REUSE) as variable_scope:
			print( "    [{}]Building scope: {}".format(self.id, variable_scope.name) )
			# input = tf.contrib.layers.maxout(inputs=input, num_units=1, axis=0)
			# input = tf.reshape(input,[1,-1])
			input = tf.layers.flatten(input)
			input = tf.layers.dense(inputs=input, units=3, activation=None, kernel_initializer=tf.initializers.variance_scaling)
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return input
			
	def _reward_prediction_loss(self):
		self.reward_prediction_labels = self._reward_prediction_target_placeholder("reward_prediction_target",1)
		return tf.reduce_sum(Categorical(self.reward_prediction_logits).cross_entropy(self.reward_prediction_labels))
		
	def sample_actions(self):
		with tf.device(self.device):
			if self.is_continuous_control():
				new_policy_batch = tf.transpose(self.policy_batch, [1, 0, 2])
				sample_batch = Normal(new_policy_batch[0], new_policy_batch[1]).sample()
				action_batch = tf.clip_by_value(sample_batch, -1,1) # Sample action batch in forward direction, use old action in backward direction
			else: # discrete control
				action_batch = Categorical(self.policy_batch).sample() # Sample action batch in forward direction, use old action in backward direction
			return action_batch
		
	def prepare_loss(self):
		with tf.device(self.device):
			print( "    [{}]Preparing loss".format(self.id) )
			# [Policy distribution]
			if self.is_continuous_control():
				# Old policy
				old_policy_batch = tf.transpose(self.old_policy_batch, [1, 0, 2])
				old_policy_distributions = Normal(old_policy_batch[0], old_policy_batch[1])
				# New policy
				new_policy_batch = tf.transpose(self.policy_batch, [1, 0, 2])
				new_policy_distributions = Normal(new_policy_batch[0], new_policy_batch[1])
			else: # discrete control
				old_policy_distributions = Categorical(self.old_policy_batch) # Old policy
				new_policy_distributions = Categorical(self.policy_batch) # New policy
			# [Actor loss]
			policy_loss_builder = PolicyLoss(
				cliprange=self.clip, 
				cross_entropy=new_policy_distributions.cross_entropy(self.old_action_batch), 
				old_cross_entropy=old_policy_distributions.cross_entropy(self.old_action_batch), 
				advantage=self.advantage_batch, 
				# entropy=self.fentropy, 
				entropy=new_policy_distributions.entropy(), 
				beta=self.beta
			)
			self.policy_loss = policy_loss_builder.get()
			# [Critic loss]
			value_loss_builder = ValueLoss(
				cliprange=self.clip, 
				value=self.value_batch, 
				old_value=self.old_value_batch, 
				reward=self.cumulative_reward_batch
			)
			self.value_loss = flags.value_coefficient * value_loss_builder.get() # usually critic has lower learning rate
			# [Extra loss]
			self.extra_loss = tf.constant(0.)
			if self.predict_reward:
				self.extra_loss += self._reward_prediction_loss()
			# [Debug variables]
			self.policy_kl_divergence = policy_loss_builder.approximate_kullback_leibler_divergence()
			self.policy_clipping_frequency = policy_loss_builder.get_clipping_frequency()
			self.policy_entropy_contribution = policy_loss_builder.get_entropy_contribution()
			self.total_loss = self.policy_loss+self.value_loss+self.extra_loss
			
	def minimize_local_loss(self, optimizer, global_step, global_var_list): # minimize loss and apply gradients to global vars.
		with tf.device(self.device) and tf.control_dependencies(self.update_keys): # control_dependencies is for batch normalization
			var_refs = [v._ref() for v in self.get_shared_keys()]
			local_gradients = tf.gradients(self.total_loss, var_refs, gate_gradients=False, aggregation_method=None, colocate_gradients_with_ops=False)
			if flags.grad_norm_clip > 0:
				local_gradients, _ = tf.clip_by_global_norm(local_gradients, flags.grad_norm_clip)
			grads_and_vars = list(zip(local_gradients, global_var_list))
			self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
			
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

	def get_shared_keys(self):
		return self.shared_keys # get model variables
		
	def predict_action(self, states, concats=None, internal_state=None):
		if internal_state is None:
			internal_state = self.lstm_default_state
		feed_dict = { self.state_batch : states, self.lstm_initial_state: internal_state }
		if self.concat_size > 0:
			feed_dict.update( { self.concat_batch : concats } )
		# return action_batch, value_batch, policy_batch, new_internal_state
		return self.session.run(fetches=[self.action_batch, self.value_batch, self.policy_batch, self.lstm_final_state], feed_dict=feed_dict)
				
	def predict_value(self, states, concats=None, internal_state=None):
		if internal_state is None:
			internal_state = self.lstm_default_state
		feed_dict = { self.state_batch : states, self.lstm_initial_state: internal_state }
		if self.concat_size > 0:
			feed_dict.update( { self.concat_batch : concats } )
		#return value_batch, new_internal_state
		return self.session.run(fetches=[self.value_batch, self.lstm_final_state], feed_dict=feed_dict)
				
	def train(self, states, actions, rewards, values, policies, discounted_cumulative_rewards, generalized_advantage_estimators, concats=None, internal_state=None, reward_prediction_states=None, reward_prediction_target=None):
		self.train_count += len(states)
		feed_dict = self.build_train_feed(states, actions, rewards, values, policies, discounted_cumulative_rewards, generalized_advantage_estimators, concats, internal_state, reward_prediction_states, reward_prediction_target)
		# run train op
		fetches=[
			self.train_op, # Minimize gradients and copy them to global network
			self.total_loss, 
			self.policy_loss, self.value_loss, self.extra_loss, 
			self.policy_kl_divergence, self.policy_clipping_frequency, self.policy_entropy_contribution
		]
		_, total_loss, policy_loss, value_loss, extra_loss, policy_kl_divergence, policy_clipping_frequency, policy_entropy_contribution = self.session.run(fetches=fetches, feed_dict=feed_dict)
		# build and return loss dict
		train_info = {"actor": policy_loss, "critic": value_loss, "actor_kl_divergence": policy_kl_divergence, "actor_clipping_frequency": policy_clipping_frequency, "actor_entropy_contribution": policy_entropy_contribution}
		if self.predict_reward:
			train_info.update( {"extra": extra_loss} )
		return total_loss, train_info
		
	def build_train_feed(self, states, actions, rewards, values, policies, discounted_cumulative_rewards, generalized_advantage_estimators, concats, internal_state, reward_prediction_states, reward_prediction_target):
		values = np.reshape(values,[-1])
		if flags.use_GAE: # Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).
			advantages = np.reshape(generalized_advantage_estimators,[-1])
			cumulative_rewards = advantages + values
		else:
			cumulative_rewards = np.reshape(discounted_cumulative_rewards,[-1])
			advantages = cumulative_rewards - values
		if internal_state is None:
			internal_state = self.lstm_default_state
		feed_dict={
				self.state_batch: states,
				self.advantage_batch: advantages,
				self.cumulative_reward_batch: cumulative_rewards,
				self.old_value_batch: values,
				self.old_policy_batch: policies,
				self.old_action_batch: actions,
				self.lstm_initial_state: internal_state # set lstm state
			}
		if self.concat_size > 0:
			feed_dict.update( {self.concat_batch : concats} )
		if self.predict_reward:
			feed_dict.update( {
				self.reward_prediction_state_batch: reward_prediction_states,
				self.reward_prediction_labels: reward_prediction_target
			} )
		return feed_dict
		
	def _lstm_default_state(self, units, batch_size=None):
		state0 = np.zeros([batch_size, units], np.float32)
		state1 = np.zeros([batch_size, units], np.float32)
		return (state0,state1)
		
	def _lstm_state_placeholder(self, units, batch_size=None, name=None):
		state0 = tf.placeholder(dtype=tf.float32, shape=[batch_size, units], name=name+"0")
		state1 = tf.placeholder(dtype=tf.float32, shape=[batch_size, units], name=name+"1")
		return (state0,state1)
		
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
		
	def _scalar_placeholder(self, name=None):
		return tf.placeholder(dtype=tf.float32, shape=(), name=name)
		
	def _concat_placeholder(self, name=None, batch_size=None):
		shape = [batch_size, self.concat_size]
		input = tf.zeros(shape if batch_size is not None else [1] + shape[1:]) # default value
		return tf.placeholder_with_default(input=input, shape=shape, name=name) # with default we can use batch normalization directly on it

	def _state_placeholder(self, name=None, batch_size=None):
		shape = [batch_size] + list(self.state_shape)
		input = tf.zeros(shape if batch_size is not None else [1] + shape[1:]) # default value
		return tf.placeholder_with_default(input=input, shape=shape, name=name) # with default we can use batch normalization directly on it