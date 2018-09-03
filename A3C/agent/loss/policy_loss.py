# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import options
flags = options.get()

class PolicyLoss(object):

	def __init__(self, cliprange, cross_entropy, old_cross_entropy, advantage, entropy, entropy_beta):
		self.cliprange = cliprange
		self.advantage = advantage
		# entropy
		self.entropy_beta = entropy_beta
		self.entropy = tf.maximum(0.,entropy) if flags.only_non_negative_entropy else entropy
		# cross entropy
		self.cross_entropy = tf.maximum(0.,cross_entropy) if flags.only_non_negative_entropy else cross_entropy
		self.old_cross_entropy = tf.maximum(0.,old_cross_entropy) if flags.only_non_negative_entropy else old_cross_entropy
		cross_entropy_shape_length = len(cross_entropy.get_shape())
		if cross_entropy_shape_length > 1:
			axis = list(range(1,cross_entropy_shape_length))
			self.cross_entropy = tf.reduce_sum(self.cross_entropy, axis)
			self.old_cross_entropy = tf.reduce_sum(self.old_cross_entropy, axis)
		# reduction function
		self.reduce_function = eval('tf.reduce_{}'.format(flags.loss_type))
		
	def get(self):
		if flags.policy_loss == 'Vanilla':
			return self.vanilla()
		elif flags.policy_loss == 'PPO':
			return self.ppo()
			
	def approximate_kullback_leibler_divergence(self): # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
		return 0.5 * self.reduce_function(tf.squared_difference(self.old_cross_entropy,self.cross_entropy))
		
	def get_clipping_frequency(self):
		return self.reduce_function(tf.to_float(tf.greater(tf.abs(tf.exp(self.old_cross_entropy - self.cross_entropy) - 1.0), self.cliprange)))
		
	def get_entropy_contribution(self):
		entropy = self.entropy
		entropy_shape_length = len(entropy.get_shape())
		if entropy_shape_length > 1:
			axis = list(range(1,entropy_shape_length))
			entropy = tf.reduce_mean(entropy, axis)
		return self.reduce_function(entropy)*self.entropy_beta
			
	def vanilla(self):
		policy = self.reduce_function(self.advantage*self.cross_entropy)
		return policy - self.get_entropy_contribution()
		
	def ppo(self):
		# Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
		ratio = tf.exp(self.old_cross_entropy - self.cross_entropy)
		clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
		min_ratio = tf.minimum(ratio, clipped_ratio)
		policy = -self.reduce_function(self.advantage*min_ratio)
		return policy - self.get_entropy_contribution()
