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
		if flags.value_loss == 'vanilla':
			return self.vanilla()
		elif flags.value_loss == 'PVO':
			return self.pvo()
		elif flags.value_loss == 'averagePVO': # used by openai
			return self.average_pvo()
			
	def vanilla(self):
		return tf.nn.l2_loss(self.reward - self.value)
		
	def pvo(self):
		# Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
		value_clipped = self.old_value + tf.clip_by_value(self.value-self.old_value, -self.cliprange, self.cliprange)
		return tf.reduce_sum( tf.maximum(tf.square(self.value - self.reward), tf.square(value_clipped - self.reward)) )
		
	def average_pvo(self):
		# Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
		value_clipped = self.old_value + tf.clip_by_value(self.value-self.old_value, -self.cliprange, self.cliprange)
		return tf.reduce_mean( tf.maximum(tf.square(self.value - self.reward), tf.square(value_clipped - self.reward)) )