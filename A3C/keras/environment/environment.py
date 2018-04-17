# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class Environment(object):
	# cached action size
	action_size = -1
	
	@staticmethod
	def create_environment(env_type, thread_index):
		if env_type == 'rogue':
			from . import rogue_environment
			return rogue_environment.RogueEnvironment(thread_index)
			
	def print_display(self):
		pass

	def __init__(self):
		pass

	def process(self, action):
		pass

	def reset(self):
		pass

	def stop(self):
		pass

	def save_episodes(self, checkpoint_dir, global_t):
		pass

	def restore_episodes(self, checkpoint_dir, global_t):
		pass
