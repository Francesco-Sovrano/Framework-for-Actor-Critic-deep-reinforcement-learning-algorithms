# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import options
flags = options.get()

class ValueLoss(object):
	def __init__(self, cliprange, value, old_value, reward):
		self.cliprange = cliprange
		self.value = value
		self.old_value = old_value
		self.reward = reward
		
	def get(self):
		if flags.value_loss == 'Vanilla':
			return self.vanilla()
		elif flags.value_loss == 'avgVanilla':
			return self.average_vanilla()
		elif flags.value_loss == 'PVO':
			return self.pvo()
		elif flags.value_loss == 'avgPVO': # used by openai
			return self.average_pvo()
			
	def vanilla(self):
		return tf.nn.l2_loss(self.reward-self.value)
		
	def average_vanilla(self):
		return 0.5*tf.reduce_mean(tf.squared_difference(self.reward, self.value))
		
	def pvo(self):
		# Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
		value_clipped = self.old_value + tf.clip_by_value(self.value-self.old_value, -self.cliprange, self.cliprange)
		max = tf.maximum(tf.abs(self.reward-self.value),tf.abs(self.reward-value_clipped))
		return tf.nn.l2_loss(max)
		
	def average_pvo(self):
		# Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
		value_clipped = self.old_value + tf.clip_by_value(self.value-self.old_value, -self.cliprange, self.cliprange)
		max = tf.maximum(tf.abs(self.reward-self.value),tf.abs(self.reward-value_clipped))
		return 0.5*tf.reduce_mean(tf.square(max))