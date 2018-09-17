# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from agent.network import BaseAC_Network

class TowersAC_Network(BaseAC_Network):
	
	# relu vs leaky_relu <https://www.reddit.com/r/MachineLearning/comments/4znzvo/what_are_the_advantages_of_relu_over_the/>
	def _cnn_layer(self, input, scope, name="", share_trainables=True):
		depth = 2
		input_shape = input.get_shape().as_list()
		with tf.variable_scope(scope), tf.variable_scope("CNN{}".format(name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "    [{}]Building scope: {}".format(self.id, variable_scope.name) )
			with tf.variable_scope("tower_1"):
				tower1 = tf.layers.conv2d(inputs=input, activation=tf.nn.relu, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_initializer=tf.initializers.variance_scaling)
				tower1 = tf.layers.conv2d(inputs=tower1, activation=tf.nn.relu, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_initializer=tf.initializers.variance_scaling)
				tower1 = tf.layers.max_pooling2d(tower1, pool_size=(input_shape[1], input_shape[2]), strides=(input_shape[1], input_shape[2]))
				tower1 = tf.layers.flatten(tower1)
			with tf.variable_scope("tower_2"):
				tower2 = tf.layers.max_pooling2d(input, pool_size=(2, 2), strides=(2, 2))
				for _ in range(depth):
					tower2 = tf.layers.conv2d(inputs=tower2, activation=tf.nn.relu, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_initializer=tf.initializers.variance_scaling)
				tower2 = tf.layers.max_pooling2d(tower2, pool_size=(input_shape[1]//2, input_shape[2]//2), strides=(input_shape[1]//2, input_shape[2]//2))
				tower2 = tf.layers.flatten(tower2)
			with tf.variable_scope("tower_3"):
				tower3 = tf.layers.max_pooling2d(input, pool_size=(4, 4), strides=(4, 4), padding='SAME')
				for _ in range(depth):
					tower3 = tf.layers.conv2d(inputs=tower3, activation=tf.nn.relu, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_initializer=tf.initializers.variance_scaling)
				tower3 = tf.layers.max_pooling2d(tower3, pool_size=(input_shape[1]//4, input_shape[2]//4), strides=(input_shape[1]//4, input_shape[2]//4))
				tower3 = tf.layers.flatten(tower3)
			concat = tf.concat([tower1, tower2, tower3], axis=-1)
			# update keys
			if share_trainables:
				self.shared_keys += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=variable_scope.name)
			self.update_keys += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=variable_scope.name)
			# return result
			return concat