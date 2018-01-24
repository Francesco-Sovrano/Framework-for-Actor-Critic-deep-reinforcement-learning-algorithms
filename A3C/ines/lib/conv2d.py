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

class Conv2D(object):

	def __init__( self, name, input, kernel, stride, depth, padding="SAME" ):
		self.name = name
		if issubclass(type(input), Conv2D):
			self.input = input.output
			self.net = input
		else:
			self.input = input
			self.net = None
		self.kernel = kernel
		self.stride = stride
		self.depth = depth
		self.padding = padding
		output = self.get_output_shape( )
		self.output = ( output[0], output[1], depth )
		
	def get_output_shape( self ):
		output = list()
		if self.padding == "SAME":
			for i in range(len(self.kernel)):
				output.append( ceil(self.input[i] / self.stride[i]) )
		else:
			for i in range(len(self.kernel)):
				output.append( ceil( (self.input[i] - self.kernel[i] + 1) / self.stride[i] ) )
		return tuple(output)
		
	def initializer(weight, height, input_channels, dtype=tf.float32):
		def _initializer(shape, dtype=dtype, partition_info=None):
			d = 1.0 / np.sqrt(input_channels * weight * height)
			return tf.random_uniform(shape, minval=-d, maxval=d)
		return _initializer
		
	def get_nodes_result(self, input_placeholder, weights, biases):
		output = tf.nn.conv2d(input_placeholder, weights, strides = [1, self.stride[0], self.stride[1], 1], padding = self.padding)
		return tf.nn.relu(output + biases)
		
	def build_variable(self, input_placeholder ):
		if self.net != None:
			input_placeholder = self.net.build_variable( input_placeholder )
			
		name_w = "weight_{0}".format(self.name)
		name_b = "bias_{0}".format(self.name)
		
		weight_shape = [self.kernel[0], self.kernel[1], self.input[2], self.depth]		
		
		w = weight_shape[0]
		h = weight_shape[1]
		input_channels	= weight_shape[2]
		output_channels = weight_shape[3]
		bias_shape = [output_channels]

		weight = tf.get_variable(name_w, weight_shape, initializer=Conv2D.initializer(w, h, input_channels))
		bias = tf.get_variable(name_b, bias_shape, initializer=Conv2D.initializer(w, h, input_channels))
		return self.get_nodes_result( input_placeholder, weight, bias )
		
	def get_flatten_size(self):
		result = 1
		for i in self.output:
			result *= i
		return result
		
	def flatten(self, tensor):
		return tf.reshape(tensor, [-1, self.get_flatten_size()])
		