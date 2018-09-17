# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import options
flags = options.get()

import numpy as np
import tensorflow as tf
from agent.network import BaseAC_Network

class NoLSTMAC_Network(BaseAC_Network):
	
	def _create_network(self):
		print( "Building network {}".format(self.id) )
		# Initialize keys collections
		self.shared_keys = []
		self.update_keys = []
		# Initialize scope names
		scope_name = "Net{0}".format(self.id)
		parent_scope_name = "Net{0}".format(self.parent.id)
		sibling_scope_name = "Net{0}".format(self.sibling.id)
		# Build nets
		with tf.device(self.device):
			# [Input]
			self.state_batch = self._state_placeholder("state")
			self.concat_batch = self._concat_placeholder("concat")
			self.old_value_batch = self._value_placeholder("old_value")
			self.old_policy_batch = self._policy_placeholder("old_policy")
			self.old_action_batch = self._action_placeholder("old_action_batch")
			self.cumulative_reward_batch = self._value_placeholder("cumulative_reward")
			self.advantage_batch = self._value_placeholder("advantage")
			# [Batch Normalization]
			# _, self.state_batch_norm = self._batch_norm_layer(input=self.state_batch, scope="Global", name="State", share_trainables=False) # global
			# [CNN]
			self.cnn = self._cnn_layer(input=self.state_batch, scope=scope_name)
			# [Concat]
			self.concat = self._concat_layer(input=self.cnn, concat=self.concat_batch, units=self.lstm_units, scope=scope_name)
			# [Policy]
			self.policy_batch = self._policy_layer(input=self.concat, scope=scope_name)
			# [Value]
			self.value_batch = self._value_layer(input=self.concat, scope=scope_name)
			# [Reward Prediction]
			if self.predict_reward:
				self.reward_prediction_state_batch = self._state_placeholder("reward_prediction_state")
				# reusing with a different placeholder seems to cause memory leaks
				reward_prediction_cnn = self._cnn_layer(input=self.reward_prediction_state_batch, scope=scope_name)
				self.reward_prediction_logits = self._reward_prediction_layer(input=reward_prediction_cnn, scope=scope_name)
		# Sample action, after getting keys
		self.action_batch = self.sample_actions()
		# Get cnn feature mean entropy
		# self.fentropy = self.get_feature_entropy(input=self.cnn, scope=scope_name)
		# Print shapes
		print( "    [{}]Input shape: {}".format(self.id, self.state_batch.get_shape()) )
		print( "    [{}]Concatenation shape: {}".format(self.id, self.concat_batch.get_shape()) )
		print( "    [{}]Tower shape: {}".format(self.id, self.cnn.get_shape()) )
		print( "    [{}]Concat shape: {}".format(self.id, self.concat.get_shape()) )
		print( "    [{}]Policy shape: {}".format(self.id, self.policy_batch.get_shape()) )
		print( "    [{}]Value shape: {}".format(self.id, self.value_batch.get_shape()) )
		if self.predict_reward:
			print( "    [{}]Reward prediction logits shape: {}".format(self.id, self.reward_prediction_logits.get_shape()) )
		print( "    [{}]Action shape: {}".format(self.id, self.action_batch.get_shape()) )
		# Prepare loss
		if self.training:
			self.prepare_loss()
			
	def predict_action(self, states, concats=None, internal_state=None):
		feed_dict = { self.state_batch : states }
		if self.concat_size > 0:
			feed_dict.update( { self.concat_batch : concats } )
		# return action_batch, value_batch, policy_batch, new_internal_state
		return self.session.run(fetches=[self.action_batch, self.value_batch, self.policy_batch, self.cnn], feed_dict=feed_dict)
				
	def predict_value(self, states, concats=None, internal_state=None):
		feed_dict = { self.state_batch : states }
		if self.concat_size > 0:
			feed_dict.update( { self.concat_batch : concats } )
		#return value_batch, new_internal_state
		return self.session.run(fetches=[self.value_batch, self.cnn], feed_dict=feed_dict)
		
	def build_train_feed(self, states, actions, rewards, values, policies, discounted_cumulative_rewards, generalized_advantage_estimators, concats, internal_state, reward_prediction_states, reward_prediction_target):
		values = np.reshape(values,[-1])
		if flags.use_GAE: # Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).
			advantages = np.reshape(generalized_advantage_estimators,[-1])
			cumulative_rewards = advantages + values
		else:
			cumulative_rewards = np.reshape(discounted_cumulative_rewards,[-1])
			advantages = cumulative_rewards - values
		feed_dict={
				self.state_batch: states,
				self.advantage_batch: advantages,
				self.cumulative_reward_batch: cumulative_rewards,
				self.old_value_batch: values,
				self.old_policy_batch: policies,
				self.old_action_batch: actions
			}
		if self.concat_size > 0:
			feed_dict.update( {self.concat_batch : concats} )
		if self.predict_reward:
			feed_dict.update( {
				self.reward_prediction_state_batch: reward_prediction_states,
				self.reward_prediction_labels: reward_prediction_target
			} )
		return feed_dict