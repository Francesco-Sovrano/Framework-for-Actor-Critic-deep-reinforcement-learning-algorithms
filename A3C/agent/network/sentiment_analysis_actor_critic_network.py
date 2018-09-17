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