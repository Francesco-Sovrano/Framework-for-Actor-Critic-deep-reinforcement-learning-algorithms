# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from agent.network import BaseAC_Network

class SAAC_Network(BaseAC_Network):
	lstm_units = 128 # the number of units of the LSTM
	
	def _cnn_layer(self, input, scope, name="", share_trainables=True):
		with tf.variable_scope(scope), tf.variable_scope("CNN{}".format(name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "    [{}]Building scope: {}".format(self.id, variable_scope.name) )
			input = tf.layers.conv2d( inputs=input, filters=16, kernel_size=(1,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return input
	
	def _concat_layer(self, input, concat, units, scope, name="", share_trainables=True):
		with tf.variable_scope(scope), tf.variable_scope("Concat{}".format(name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "    [{}]Building scope: {}".format(self.id, variable_scope.name) )
			input = tf.layers.flatten(input)
			input = tf.layers.dense(inputs=input, units=2*units, activation=None, kernel_initializer=tf.initializers.variance_scaling)
			input = tf.contrib.layers.maxout(inputs=input, num_units=units, axis=-1)
			input = tf.reshape(input, [-1, units])
			if concat.get_shape()[-1] > 0:
				concat = tf.layers.flatten(concat)
				input = tf.concat([input, concat], -1) # shape: (batch, concat_size+units)
			# Update keys
			self._update_keys(variable_scope.name, share_trainables)
			# Return result
			return input
			
	# no dropout!
	def _lstm_layer(self, input, initial_state, scope, name="", share_trainables=True):
		with tf.variable_scope(scope), tf.variable_scope("LSTM{}".format(name), reuse=tf.AUTO_REUSE) as variable_scope:
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
			lstm_outputs = tf.reshape(lstm_outputs, [-1,units]) # shape: (batch, units)
			# Update keys
			self._update_keys(variable_scope.name, share_trainables)
			# Return result
			return lstm_outputs, final_state