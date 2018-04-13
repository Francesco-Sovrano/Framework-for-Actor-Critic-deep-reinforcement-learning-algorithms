# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from model.a3c_model import A3CModel

class MultiAgentModel(object):
	def __init__( self, id, state_shape, agents_count, action_size, entropy_beta, device ):
		self._id = id
		self._device = device
		self.action_size = action_size
		# input size
		self.agent_count = agents_count
		self._agent_state_shape = state_shape
		self._agent_list = []
		# create networks
		for i in range(self.agent_count):
			self._agent_list.append ( A3CModel( str(id)+"_"+str(i), self._agent_state_shape, action_size, entropy_beta, device ) )

	def get_agent( self, id ):
		return self._agent_list[id]
		
	def get_vars(self):
		vars = []
		for agent in self._agent_list:
			vars = set().union(agent.get_vars(),vars)
		return list(vars)
		
	def reset(self):
		for agent in self._agent_list:
			agent.reset_state()
		
	def concat_action_and_reward(self, action, reward):
		"""
		Return one hot vectored action and reward.
		"""
		action_reward = np.zeros([self.action_size+1])
		action_reward[action] = 1.0
		action_reward[-1] = float(reward)
		return action_reward