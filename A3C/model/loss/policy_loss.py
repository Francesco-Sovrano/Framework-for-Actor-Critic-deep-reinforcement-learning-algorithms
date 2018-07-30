# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import options
flags = options.get()

from model.loss.loss import Loss

class PolicyLoss(Loss):
		
	def get(self):
		if flags.policy_loss == 'vanilla':
			return self.vanilla()
		elif flags.policy_loss == 'PPO':
			return self.ppo()
		elif flags.policy_loss == 'averagePPO':
			return self.average_ppo()
		elif flags.policy_loss == 'openaiPPO': # openAI implementation
			return self.openai_ppo()
			
	def get_log_policy(self, policy):
		return tf.log(tf.clip_by_value(policy, 1e-20, 1.0)) # Avoid NaN with clipping when value in pi becomes zero
	
	def get_action_log_policy(self, log_policy, action):
		return tf.reduce_sum(tf.multiply(log_policy, action), reduction_indices=1 if len(self.policy.get_shape()) < 3 else [1,2])
		
	def vanilla(self):
		# Avoid NaN with clipping when value in pi becomes zero
		log_policy = self.get_log_policy(self.policy)
		# Policy entropy
		entropy = -tf.reduce_sum(self.policy * log_policy, reduction_indices=1 if len(self.policy.get_shape()) < 3 else [1,2])
		# Policy loss (output)
		return -tf.reduce_sum( self.get_action_log_policy(log_policy, self.action) * self.advantage + entropy * self.entropy_beta )
		
	def ppo(self):
		# Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
		# old
		old_log_policy = self.get_log_policy(self.old_policy)
		old_action_log_policy = self.get_action_log_policy(old_log_policy, self.action)
		# new
		log_policy = self.get_log_policy(self.policy)
		action_log_policy = self.get_action_log_policy(log_policy, self.action)
		# ratio
		ratio = tf.exp(action_log_policy - old_action_log_policy)
		clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
		# entropy
		entropy = -tf.reduce_sum(self.policy * log_policy, reduction_indices=1 if len(self.policy.get_shape()) < 3 else [1,2])
		# Policy loss
		return -tf.reduce_sum( tf.minimum(ratio, clipped_ratio) * self.advantage + entropy * self.entropy_beta )
				
	def average_ppo(self):
		# Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
		# old
		old_log_policy = self.get_log_policy(self.old_policy)
		old_action_log_policy = self.get_action_log_policy(old_log_policy, self.action)
		# new
		log_policy = self.get_log_policy(self.policy)
		action_log_policy = self.get_action_log_policy(log_policy, self.action)
		# ratio
		ratio = tf.exp(action_log_policy - old_action_log_policy)
		clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
		# entropy
		entropy = -tf.reduce_sum(self.policy * log_policy, reduction_indices=1 if len(self.policy.get_shape()) < 3 else [1,2])
		# Policy loss
		# self.advantage = (self.advantage - tf.reduce_mean(self.advantage)) / (self.reduce_std(self.advantage) + 1e-8)
		return -tf.reduce_mean( tf.minimum(ratio, clipped_ratio) * self.advantage + entropy * self.entropy_beta )
		
	def openai_ppo(self):
		# Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
		def log_pac(policy, action):
			mean = tf.reduce_mean(input_tensor=policy, axis=1, keepdims=True)
			std = self.reduce_std(input_tensor=policy, axis=1, keepdims=True)
			logstd = tf.log(std)
			return 0.5 * tf.reduce_sum(tf.square((action - mean) / std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(action)[-1]) \
               + tf.reduce_sum(logstd, axis=-1)
		# old
		old_action_log_policy = log_pac(self.old_policy, self.action)
		# new
		action_log_policy = log_pac(self.policy, self.action)
		# ratio
		ratio = tf.exp(action_log_policy - old_action_log_policy)
		clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
		# entropy
		entropy = -tf.reduce_sum(tf.log(self.reduce_std(input_tensor=self.policy, axis=1)) + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
		# Policy loss
		return -tf.reduce_mean( tf.minimum(ratio, clipped_ratio) * self.advantage + entropy * self.entropy_beta )