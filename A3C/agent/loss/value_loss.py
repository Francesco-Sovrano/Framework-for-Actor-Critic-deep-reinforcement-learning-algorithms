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
		self.reduce_function = eval('tf.reduce_{}'.format(flags.loss_type))
		
	def get(self):
		if flags.value_loss == 'Vanilla':
			return self.vanilla()
		elif flags.value_loss == 'PVO':
			return self.pvo()
			
	def vanilla(self):
		return 0.5*self.reduce_function(tf.squared_difference(self.reward, self.value))
				
	def pvo(self):
		# Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
		value_clipped = self.old_value + tf.clip_by_value(self.value-self.old_value, -self.cliprange, self.cliprange)
		max = tf.maximum(tf.abs(self.reward-self.value),tf.abs(self.reward-value_clipped))
		return 0.5*self.reduce_function(tf.square(max))