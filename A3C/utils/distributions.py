# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# def separate(self, input, value):
	# input_shape = tf.shape(input)
	# true_labels = tf.ones(input_shape)
	# false_labels = tf.zeros(input_shape)
	# mask = tf.where(tf.greater_equal(input, value), true_labels, false_labels)
	# greater_equal = mask*input
	# lower = input - greater_equal
	# return greater_equal, lower
	
class Categorical(object):
	
	def __init__(self, logits):
		self.logits = logits
	
	def cross_entropy(self, samples):
		return tf.nn.softmax_cross_entropy_with_logits_v2(labels=samples, logits=self.logits)

	def entropy(self):
		a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
		ea0 = tf.exp(a0)
		z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
		p0 = ea0 / z0
		return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)
			
	def sample(self):
		logits_shape = self.logits.get_shape()
		u = tf.random_uniform(tf.shape(self.logits))
		samples = tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)
		depth = logits_shape.as_list()[-1]
		one_hot_actions = tf.one_hot(samples, depth)
		one_hot_actions.set_shape(logits_shape)
		return one_hot_actions
		
class Normal(object):

	def __init__(self, mean, std):
		self.distribution = tf.distributions.Normal(mean, std, validate_args=False) # validate_args is computationally expensive
	
	def cross_entropy(self, samples):
		return -self.distribution.log_prob(samples) # probability density function

	def entropy(self):
		return self.distribution.entropy()
			
	def sample(self):
		return self.distribution.sample()