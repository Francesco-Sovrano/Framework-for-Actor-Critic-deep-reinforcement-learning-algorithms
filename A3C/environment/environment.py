# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class Environment(object):
	@staticmethod
	def create_environment(env_type, thread_index):
		if env_type == 'rogue':
			from . import rogue_environment
			return rogue_environment.RogueEnvironment(thread_index)
		elif env_type == 'car_controller':
			from . import car_controller_environment
			return car_controller_environment.CarControllerEnvironment(thread_index)
		else:
			from . import gym_environment
			return gym_environment.GymEnvironment(thread_index,env_type)
			
	def choose_action(self, pi_values):
		return np.random.choice(range(len(pi_values)), p=pi_values)
		
	def get_last_action_reward(self):
		action_reward = np.zeros(self.get_action_size()+1)
		action_reward[self.last_action]=1
		action_reward[-1] = self.last_reward
		return action_reward

	def __init__(self):
		pass

	def process(self, action):
		pass

	def reset(self):
		pass

	def stop(self):
		pass
		
	def get_state_shape(self):
		pass
		
	def get_screen_shape(self):
		return self.get_state_shape()
