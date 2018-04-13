# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

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
		# lstm
		self._lstm_units = 256
		# init
		self._create_network()
	
	def _create_network(self):
		scope_name = "net_{0}".format(self._id)
		with tf.device(self._device), tf.variable_scope(scope_name) as scope:
			# [LSTM]
			self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self._lstm_units, state_is_tuple=True)
			# [base network]
			self._create_base_network()
			
			self.reset_state()
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
			
	def _create_base_network(self):
		self.base_input = self._empty_input_placeholder()
		self.base_last_action_reward_input = self._action_reward_input_placeholder() # Last action and reward
		
		# Conv layers
		conv_result = self._conv_result(self.base_input)
		# LSTM layer
		self.base_initial_lstm_state0 = self._lstm_state_placeholder()
		self.base_initial_lstm_state1 = self._lstm_state_placeholder()
		
		self.base_initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.base_initial_lstm_state0, self.base_initial_lstm_state1)

		self.base_lstm_outputs, self.base_lstm_state = self._base_lstm_layer(conv_result, self.base_last_action_reward_input, self.base_initial_lstm_state)

		self.base_pi = self._base_policy_layer(self.base_lstm_outputs) # policy output
		self.base_value	= self._base_value_layer(self.base_lstm_outputs)	# value output

	def _conv_result(self, input, reuse=False):
		with tf.variable_scope("base_conv{0}".format(self._id), reuse=reuse) as scope:
			input = tf.layers.conv2d( inputs=input, filters=16, kernel_size=(3,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )
			input = tf.layers.conv2d( inputs=input, filters=32, kernel_size=(3,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )
			return input
	
	def _base_lstm_layer(self, input, last_action_reward_input, initial_state_input, reuse=False):
		with tf.variable_scope("base_lstm{0}".format(self._id), reuse=reuse) as scope:
		
			input = tf.layers.flatten(input) # input shape: (batch,w*h*depth)
			input = tf.layers.dense( inputs=input, units=self._lstm_units, activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )
			
			step_size = tf.shape(input)[:1] # (unroll_step, 256)
			input = tf.concat([input, last_action_reward_input], 1) # (unroll_step, 256+action_size+1)
			input = tf.reshape(input, [1, -1, self._lstm_units+self._action_size+1]) # (1, unroll_step, 256+action_size+1)

			lstm_outputs, lstm_state = tf.nn.dynamic_rnn(self.lstm_cell, input, initial_state = initial_state_input, sequence_length = step_size, time_major = False, scope = scope)
			lstm_outputs = tf.reshape(lstm_outputs, [-1,self._lstm_units]) #(1,unroll_step,256) for back prop, (1,1,256) for forward prop.
			return lstm_outputs, lstm_state

	def _base_policy_layer(self, input, reuse=False):
		with tf.variable_scope("base_policy{0}".format(self._id), reuse=reuse) as scope:
			# Policy (output)
			input = tf.layers.dense( inputs=input, units=self._action_size, activation=None, kernel_initializer=tf.initializers.variance_scaling )
			return tf.nn.softmax( input )

	def _base_value_layer(self, input, reuse=False):
		with tf.variable_scope("base_value{0}".format(self._id), reuse=reuse) as scope:
			# Value (output)
			input = tf.layers.dense( inputs=input, units=1, activation=None, kernel_initializer=tf.initializers.variance_scaling )
			return tf.reshape(input, [-1]) # flatten output

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
		self.base_lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, self._lstm_units]), np.zeros([1, self._lstm_units]))

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
			
	def _empty_input_placeholder(self):
		return tf.placeholder(tf.float32, np.concatenate([[None], self._state_shape], 0))
		
	def _action_reward_input_placeholder(self):
		return tf.placeholder(tf.float32, [None, self._action_size+1])
		
	def _lstm_state_placeholder(self):
		return tf.placeholder(tf.float32, [1, self._lstm_units])
		