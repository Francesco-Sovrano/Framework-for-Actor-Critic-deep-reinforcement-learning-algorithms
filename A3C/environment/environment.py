# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import options
flags = options.get()

class Environment(object):
	@staticmethod
	def create_environment(env_type, thread_index, training):
		if env_type == 'rogue':
			from . import rogue_environment
			return rogue_environment.RogueEnvironment(thread_index)
		elif env_type == 'car_controller':
			from . import car_controller_environment
			return car_controller_environment.CarControllerEnvironment(thread_index)
		elif env_type == 'sentipolc':
			from . import sentipolc_environment
			return sentipolc_environment.SentiPolcEnvironment(thread_index, training)
		else:
			from . import gym_environment
			return gym_environment.GymEnvironment(thread_index, env_type)
		
	def get_concatenation_size(self):
		return np.prod(self.get_action_shape())+1
		
	def get_concatenation(self):
		return np.concatenate((self.last_action,[self.last_reward]), -1)
		
	def choose_action(self, action_vector):
		return np.argwhere(action_vector==1)[0][0]

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
		
	def get_test_result(self):
		return None
		
	def get_test_size(self):
		return flags.match_count_for_evaluation//flags.parallel_size
		
	def evaluate_test_results(self, test_result_file):
		pass
		
	def get_screen_shape(self):
		return self.get_state_shape()
