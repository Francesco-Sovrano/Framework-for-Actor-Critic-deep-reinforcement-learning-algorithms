# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import options
flags = options.get()

class PolicyLoss(object):

	def __init__(self, cliprange, cross_entropy, old_cross_entropy, advantage, entropy, beta):
		self.cliprange = cliprange
		self.advantage = advantage
		# entropy
		self.beta = beta
		self.entropy = tf.maximum(0.,entropy) if flags.only_non_negative_entropy else entropy
		self.cross_entropy = tf.maximum(0.,cross_entropy) if flags.only_non_negative_entropy else cross_entropy
		self.old_cross_entropy = tf.maximum(0.,old_cross_entropy) if flags.only_non_negative_entropy else old_cross_entropy
		# sum entropies in case the agent has to predict more than one action
		if len(self.cross_entropy.get_shape()) > 1:
			self.cross_entropy = tf.reduce_sum(self.cross_entropy, -1)
		if len(self.old_cross_entropy.get_shape()) > 1:
			self.old_cross_entropy = tf.reduce_sum(self.old_cross_entropy, -1)
		if len(self.entropy.get_shape()) > 1:
			self.entropy = tf.reduce_sum(self.entropy, -1)
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
		return tf.reduce_mean(tf.to_float(tf.greater(tf.abs(tf.exp(self.old_cross_entropy - self.cross_entropy) - 1.0), self.cliprange)))
		
	def get_entropy_contribution(self):
		return self.reduce_function(self.entropy)*self.beta
			
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
