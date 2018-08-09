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
		self.cross_entropy = cross_entropy
		self.old_cross_entropy = old_cross_entropy
		self.advantage = advantage
		self.entropy = entropy
		self.entropy_beta = entropy_beta
		
	def get(self):
		if flags.policy_loss == 'vanilla':
			return self.vanilla()
		elif flags.policy_loss == 'PPO':
			return self.ppo()
		elif flags.policy_loss == 'averagePPO':
			return self.average_ppo()
			
	def vanilla(self):
		policy = tf.reduce_sum(self.cross_entropy*self.advantage)
		entropy = tf.reduce_sum(self.entropy)*self.entropy_beta
		return policy - entropy
		
	def ppo(self):
		# Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
		ratio = tf.exp(self.old_cross_entropy - self.cross_entropy)
		clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
		policy = tf.reduce_sum(-tf.minimum(ratio, clipped_ratio)*self.advantage)
		entropy = tf.reduce_sum(self.entropy)*self.entropy_beta
		return policy - entropy
				
	def average_ppo(self):
		# Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
		ratio = tf.exp(self.old_cross_entropy - self.cross_entropy)
		clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
		policy = tf.reduce_mean(-tf.minimum(ratio, clipped_ratio)*self.advantage)
		entropy = tf.reduce_mean(self.entropy)*self.entropy_beta
		return policy - entropy