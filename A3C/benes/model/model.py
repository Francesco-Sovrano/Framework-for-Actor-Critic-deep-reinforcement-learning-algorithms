# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import misc
from skimage import exposure
from math import ceil
from lib.conv2d import Conv2D
from lib.maxpool import MaxPool

# weight initialization based on muupan's code
# https://github.com/muupan/async-rl/blob/master/a3c_ale.py
def fc_initializer(input_channels, dtype=tf.float32):
	def _initializer(shape, dtype=dtype, partition_info=None):
		d = 1.0 / np.sqrt(input_channels)
		return tf.random_uniform(shape, minval=-d, maxval=d)
	return _initializer

class MultiTowerModel(object):
	def __init__( self, id, state_shape, action_size, entropy_beta, device ):
		self._id = id
		self._device = device
		self.action_size = action_size
		# input size
		self._tower_count = state_shape[0]
		self._tower_state_shape = (state_shape[1], state_shape[2], 1)
		self._tower_list = []
		# create networks
		for i in range(self._tower_count):
			self._tower_list.append ( A3CModel( str(id)+"_"+str(i), self._tower_state_shape, action_size, entropy_beta, device ) )
		# get variables
		self._variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) # returns variables for all the towers

	def get_best_tower_and_layer( self, state ):
		environment = state["environment"]
		layer = state["layer"]
		return self._tower_list[layer], np.expand_dims(environment[layer],axis=-1), layer
		
	def get_vars(self):
		return self._variables
		
	def reset(self):
		for tower in self._tower_list:
			tower.reset_state()
		
	def concat_action_and_reward(self, action, reward):
		"""
		Return one hot vectored action and reward.
		"""
		action_reward = np.zeros([self.action_size+1])
		action_reward[action] = 1.0
		action_reward[-1] = float(reward)
		return action_reward

class A3CModel(object):
	def get_vars(self):
		return self.variables
		
	def __init__( self, id, state_shape, action_size, entropy_beta, device ):
		self._id = id
		self._device = device
		self._action_size = action_size
		self._entropy_beta = entropy_beta
		# input size
		self._state_shape = state_shape
		# base layer
		self._tower1_conv1 = Conv2D( name = "tower1_conv1", input = state_shape, kernel = (3,3), stride = (1,1), depth = 16 )
		self._tower1 = Conv2D( name = "tower1_conv2", input = self._tower1_conv1, kernel = (3,3), stride = (1,1), depth = 32 )
		# lstm
		self._lstm_output_size = 256
		# value layer
		self._value_layer_shape = (self._lstm_output_size, 1, 1)
		self._value_conv1 = Conv2D( name = "value_conv1", input = self._value_layer_shape, kernel = (3,3), stride = (1,1), depth = 16 )
		self._value = Conv2D( name = "value_conv2", input = self._value_conv1, kernel = (3,3), stride = (1,1), depth = 32 )
		# policy layer
		self._policy_layer_shape = (self._lstm_output_size, 1, 1)
		self._policy_conv1 = Conv2D( name = "policy_conv1", input = self._policy_layer_shape, kernel = (3,3), stride = (1,1), depth = 16 )
		self._policy = Conv2D( name = "policy_conv2", input = self._policy_conv1, kernel = (3,3), stride = (1,1), depth = 32 )
		# init
		self._create_network()
	
	def _create_network(self):
		scope_name = "net_{0}".format(self._id)
		with tf.device(self._device), tf.variable_scope(scope_name) as scope:
			# [LSTM]
			self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self._lstm_output_size, state_is_tuple=True)
			# [base network]
			self._create_base_network()
			
			self.reset_state()
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
			
	def _create_base_network(self):
		self.base_input = self._empty_input_placeholder()
		self.base_last_action_reward_input = self._action_reward_input_placeholder() # Last action and reward
		
		# Conv layers
		conv_result, conv_size = self._conv_result()
		# LSTM layer
		self.base_initial_lstm_state0 = self._lstm_state_placeholder()
		self.base_initial_lstm_state1 = self._lstm_state_placeholder()
		
		self.base_initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.base_initial_lstm_state0, self.base_initial_lstm_state1)

		self.base_lstm_outputs, self.base_lstm_state = self._base_lstm_layer(conv_result, conv_size, self.base_last_action_reward_input, self.base_initial_lstm_state)

		self.base_pi = self._base_policy_layer(self.base_lstm_outputs) # policy output
		self.base_value	= self._base_value_layer(self.base_lstm_outputs)	# value output

	def _conv_result(self, reuse=False):
		with tf.variable_scope("base_conv{0}".format(self._id), reuse=reuse) as scope:
		
			result = self._tower1.build_variable(self.base_input)
			flatten_size = self._tower1.get_flatten_size()
			return result, flatten_size
	
	def _base_lstm_layer(self, conv_result, conv_size, last_action_reward_input, initial_state_input, reuse=False):
		with tf.variable_scope("base_lstm{0}".format(self._id), reuse=reuse) as scope:
		
			weights_fc1, biases_fc1 = self._fc_variable( [conv_size, self._lstm_output_size], "base_fc1" )
			conv_output_flat = tf.reshape(conv_result, [-1, int(conv_size)])
			conv_output_fc = tf.nn.relu(tf.matmul(conv_output_flat, weights_fc1) + biases_fc1)
			
			step_size = tf.shape(conv_output_fc)[:1] # (unroll_step, 256)
			lstm_input = tf.concat([conv_output_fc, last_action_reward_input], 1) # (unroll_step, 256+action_size+1)
			lstm_input_reshaped = tf.reshape(lstm_input, [1, -1, self._lstm_output_size+self._action_size+1]) # (1, unroll_step, 256+action_size+1)

			lstm_outputs, lstm_state = tf.nn.dynamic_rnn(self.lstm_cell, lstm_input_reshaped, initial_state = initial_state_input, sequence_length = step_size, time_major = False, scope = scope)
			lstm_outputs = tf.reshape(lstm_outputs, [-1,self._lstm_output_size]) #(1,unroll_step,256) for back prop, (1,1,256) for forward prop.
			return lstm_outputs, lstm_state

	def _base_policy_layer(self, lstm_outputs, reuse=False):
		with tf.variable_scope("base_policy{0}".format(self._id), reuse=reuse) as scope:
			layer_input = tf.expand_dims( tf.expand_dims( lstm_outputs, -1 ), -1 ) #(unroll_step,256,1,1)
			result = self._policy.build_variable( layer_input ) # Weight for value output layer
			flatten_result = self._policy.flatten(result)
			
			# Policy (output)
			W_fc_p, b_fc_p = self._fc_variable([self._policy.get_flatten_size(), self._action_size], "base_fc_p")
			return tf.nn.softmax( tf.matmul(flatten_result, W_fc_p) + b_fc_p )

	def _base_value_layer(self, lstm_outputs, reuse=False):
		with tf.variable_scope("base_value{0}".format(self._id), reuse=reuse) as scope:
			layer_input = tf.expand_dims( tf.expand_dims( lstm_outputs, -1 ), -1 ) #(unroll_step,256,1,1)
			result = self._value.build_variable( layer_input ) # Weight for value output layer
			flatten_result = self._value.flatten(result)
			
			# Value (output)
			W_fc_v, b_fc_v = self._fc_variable([self._value.get_flatten_size(), 1], "base_fc_v")
			return tf.reshape(tf.matmul(flatten_result, W_fc_v) + b_fc_v,[-1]) # flatten output

	def _base_loss(self):
		# [base A3C]
		# Taken action (input for policy)
		self.base_a = tf.placeholder(tf.float32, [None, self._action_size])
		
		# Advantage (R-V) (input for policy)
		self.base_adv = tf.placeholder(tf.float32, [None])
		
		# Avoid NaN with clipping when value in pi becomes zero
		log_pi = tf.log(tf.clip_by_value(self.base_pi, 1e-20, 1.0))
		
		# Policy entropy
		entropy = -tf.reduce_sum(self.base_pi * log_pi, reduction_indices=1)
		
		# Policy loss (output)
		policy_loss = -tf.reduce_sum( tf.reduce_sum( tf.multiply( log_pi, self.base_a ), reduction_indices=1 ) * self.base_adv + entropy * self._entropy_beta )
		
		# R (input for value target)
		self.base_r = tf.placeholder(tf.float32, [None])
		
		# Value loss (output)
		# (Learning rate for Critic is half of Actor's, so multiply by 0.5)
		value_loss = 0.5 * tf.nn.l2_loss(self.base_r - self.base_value)
		
		base_loss = policy_loss + value_loss
		return base_loss

	def prepare_loss(self):
		with tf.device(self._device):
			self.total_loss = self._base_loss()

	def reset_state(self):
		self.base_lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, self._lstm_output_size]), np.zeros([1, self._lstm_output_size]))

	def run_policy_and_value(self, sess, s_t, last_action_reward):
		# This run_policy_and_value() is used when forward propagating.
		# so the step size is 1.
		pi_out, v_out, self.base_lstm_state_out = sess.run( [self.base_pi, self.base_value, self.base_lstm_state], feed_dict = {self.base_input : [s_t], self.base_last_action_reward_input : [last_action_reward], self.base_initial_lstm_state0 : self.base_lstm_state_out[0], self.base_initial_lstm_state1 : self.base_lstm_state_out[1]} )
		# pi_out: (1,3), v_out: (1)
		return (pi_out[0], v_out[0])
		
	def run_value(self, sess, s_t, last_action_reward):
		# This run_bae_value() is used for calculating V for bootstrapping at the 
		# end of LOCAL_T_MAX time step sequence.
		# When next sequence starts, V will be calculated again with the same state using updated network weights,
		# so we don't update LSTM state here.
		v_out, _ = sess.run( [self.base_value, self.base_lstm_state], feed_dict = {self.base_input : [s_t], self.base_last_action_reward_input : [last_action_reward], self.base_initial_lstm_state0 : self.base_lstm_state_out[0], self.base_initial_lstm_state1 : self.base_lstm_state_out[1]} )
		return v_out[0]

	def sync_from(self, src_network, name=None):
		src_vars = src_network.get_vars()
		dst_vars = self.get_vars()

		sync_ops = []

		with tf.device(self._device):
			with tf.name_scope(name, "A3CModel{0}".format(self._id),[]) as name:
				for(src_var, dst_var) in zip(src_vars, dst_vars):
					sync_op = tf.assign(dst_var, src_var)
					sync_ops.append(sync_op)

				return tf.group(*sync_ops, name=name)
			
	def _fc_variable(self, weight_shape, name):
		name_w = "W_{0}{1}".format(name, self._id)
		name_b = "b_{0}{1}".format(name, self._id)
		
		input_channels	= weight_shape[0]
		output_channels = weight_shape[1]
		bias_shape = [output_channels]

		weight = tf.get_variable(name_w, weight_shape, initializer=fc_initializer(input_channels))
		bias = tf.get_variable(name_b, bias_shape, initializer=fc_initializer(input_channels))
		return weight, bias

	def _empty_input_placeholder(self):
		return tf.placeholder(tf.float32, np.concatenate([[None], self._state_shape], 0))
		
	def _action_reward_input_placeholder(self):
		return tf.placeholder(tf.float32, [None, self._action_size+1])
		
	def _lstm_state_placeholder(self):
		return tf.placeholder(tf.float32, [1, self._lstm_output_size])
		