# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from agent.network import BaseAC_Network

class HybridTowersAC_Network(BaseAC_Network):
	
	# relu vs leaky_relu <https://www.reddit.com/r/MachineLearning/comments/4znzvo/what_are_the_advantages_of_relu_over_the/>
	def _cnn_layer(self, input, scope, name="", share_trainables=True):
		input_shape = input.get_shape().as_list()
		with tf.variable_scope(scope), tf.variable_scope("CNN{}".format(name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "    [{}]Building scope: {}".format(self.id, variable_scope.name) )
			tower1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_initializer=tf.initializers.variance_scaling)
			tower1 = tf.layers.conv2d(inputs=tower1, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_initializer=tf.initializers.variance_scaling)
			tower1 = tf.layers.max_pooling2d(tower1, pool_size=(input_shape[1], input_shape[2]), strides=(input_shape[1], input_shape[2]))
			tower1 = tf.layers.flatten(tower1)
			input = tf.layers.conv2d(inputs=input, filters=16, kernel_size=(3,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )
			input = tf.layers.conv2d(inputs=input, filters=8, kernel_size=(3,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )
			input = tf.layers.flatten(input)
			concat = tf.concat([tower1, input], axis=-1)
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return concat