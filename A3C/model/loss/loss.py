# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Loss(object):
	def __init__(self, cliprange, policy, old_policy, value, old_value, action, advantage, reward, entropy_beta):
		self.cliprange = cliprange
		self.policy = policy
		self.old_policy = old_policy
		self.value = value
		self.old_value = old_value
		self.action = action
		self.advantage = advantage
		self.reward = reward
		self.entropy_beta = entropy_beta
		
	def reduce_std(self, input_tensor, axis=None, keepdims=False): # standard deviation
		return tf.sqrt(tf.reduce_mean(tf.square(input_tensor - tf.reduce_mean(input_tensor, keepdims=True)), axis, keepdims))