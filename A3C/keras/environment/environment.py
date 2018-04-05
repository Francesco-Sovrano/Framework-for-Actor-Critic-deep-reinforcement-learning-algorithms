# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod


class Environment(ABC):
	# cached action size
	action_size = -1

	@staticmethod
	def create_environment(env_type, thread_index):
		"""Returns an environment of the given type

		:param str env_type:
			environment type
		:param int thread_index:
			thead index
		:rtype: Environment
		:return:
			selected environment
		"""
		if env_type == 'rogue':
			from . import rogue_environment
			return rogue_environment.RogueEnvironment(thread_index)
		raise ValueError('unknown environment type "%s"' % env_type)

	def __init__(self):
		self._situation_generator = self._create_situation_generator()
		self.last_state = None
		self.last_action = None
		self.last_reward = None
		self.last_situation = None

	@staticmethod
	def _instantiate_from_module(module, cls_name, on_exc_return_name=True):
		"""Returns an instance of a class from a module.
		If "on_exc_return_name" is True and there's no such class in the module, returns "cls_name"

		:param module:
			module where the class is located
		:param str cls_name:
			name of the class
		:param bool on_exc_return_name:
			whether tu return "cls_name" if there is no class with such name
		:return:
			an instance of "cls_name"
		"""
		if not hasattr(module, cls_name):
			if on_exc_return_name:
				return cls_name
			raise ValueError('no class named "%s" was found in module %s' % (cls_name, module.__name__))
		return getattr(module, cls_name)()

	@abstractmethod
	def _create_situation_generator(self):
		"""Creates and returns a situation generator

		:rtype: situations.SituationGenerator
		"""
		pass

	def get_situation_generator(self):
		"""Returns the environment's situation generator

		:rtype: SituationGenerator
		"""
		return self._situation_generator

	def get_situations_count(self):
		return self._situation_generator.situations_count()

	@abstractmethod
	def get_action_size(self):
		pass

	@abstractmethod
	def get_state_shape(self):
		pass

	@abstractmethod
	def print_display(self):
		pass

	@abstractmethod
	def process(self, action):
		pass

	def reset(self):
		self._situation_generator.reset()

	@abstractmethod
	def stop(self):
		pass
