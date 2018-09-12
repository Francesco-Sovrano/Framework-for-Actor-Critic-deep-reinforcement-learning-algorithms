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
		
	def probability_distribution(self):
		return tf.nn.softmax(self.logits)
	
	def cross_entropy(self, samples):
		return tf.nn.softmax_cross_entropy_with_logits_v2(labels=samples, logits=self.logits)

	def entropy(self):
		scaled_logits = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
		exp_scaled_logits = tf.exp(scaled_logits)
		sum_exp_scaled_logits = tf.reduce_sum(exp_scaled_logits, axis=-1, keepdims=True)
		avg_exp_scaled_logits = exp_scaled_logits / sum_exp_scaled_logits
		return tf.reduce_sum(avg_exp_scaled_logits * (tf.log(sum_exp_scaled_logits) - scaled_logits), axis=-1)
			
	def sample(self):
		scaled_logits = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
		samples = tf.squeeze(tf.multinomial(scaled_logits, 1), axis=-1) # one sample per batch
		depth = self.logits.get_shape().as_list()[-1] # depth of the one hot vector
		return tf.one_hot(indices=samples, depth=depth) # one_hot_actions
		
class Normal(object):

	def __init__(self, mean, std):
		self.distribution = tf.distributions.Normal(mean, std, validate_args=False) # validate_args is computationally expensive
	
	def cross_entropy(self, samples):
		return -self.distribution.log_prob(samples) # probability density function

	def entropy(self):
		return self.distribution.entropy()
			
	def sample(self):
		return self.distribution.sample()