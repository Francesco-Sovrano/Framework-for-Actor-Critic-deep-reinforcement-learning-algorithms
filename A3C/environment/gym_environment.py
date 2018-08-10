# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# get command line args
import options
flags = options.get()

import gym
from environment import environment
import collections
import numpy as np
        
class GymEnvironment(environment.Environment):
	__slots__ = ( 'thread_index','real_actions','game','last_action','last_reward','last_state','episodes','use_ram','cumulative_reward','step' )
	
	def __init__(self, thread_index, environment_name):
		environment.Environment.__init__(self)
		self.thread_index = thread_index
		# setup environment
		self.game = gym.make(environment_name)
		self.real_actions = self.game.action_space
		# evaluator stuff
		self.episodes = collections.deque()
		self.use_ram = "-ram" in environment_name

	def reset(self):
		self.stop()
		self.last_state = self.game.reset()
		self.last_state = self.normalize(self.last_state)
		self.last_action = 0
		self.last_reward = 0
		self.cumulative_reward = 0
		self.step = 0
		
	def normalize(self, state):
		while len(state.shape) < 3:
			state = np.expand_dims(state, axis=-1)
		return state
			
	def get_action_shape(self):
		return (self.real_actions.n,1)
		
	def get_state_shape(self):
		shape = self.game.observation_space.shape
		while len(shape) < 3:
			shape = shape + (1,)
		return shape
		
	def stop(self):
		self.game.close()
		
	def get_statistics(self):
		result = {}
		result["avg_reward"] = 0
		result["avg_steps"] = 0
		count = len(self.episodes)
		if count>0:
			for e in self.episodes:
				result["avg_steps"] += e["step"]
				result["avg_reward"] += e["reward"]
			result["avg_steps"] /= count
			result["avg_reward"] /= count
		return result
		
	def get_screen(self):
		return self.last_state
		
	def get_frame_info(self, network, observation, value, action, reward, cross_entropy):
		state_info = "reward={}, agent={}, value={}, cross_entropy={}\n".format(reward, network.agent_id, value, cross_entropy)
		action_info = "action={}\n".format(action)
		frame_info = { "log": state_info + action_info }
		if flags.save_episode_screen:
			if self.use_ram: # ram
				observation_info = "observation={}\n".format(np.array_str(observation.flatten()))
				frame_info["log"] += observation_info
				frame_info["screen"] = { "value": frame_info["log"], "type": 'ASCII' }
			else: # rgb image
				frame_info["screen"] = { "value": observation, "type": 'RGB' }
		return frame_info
		
	def process(self, action_vector):
		action = np.argwhere(action_vector==1)[0][0]
		# self.game.render(mode='rgb_array')
		state, reward, terminal, info = self.game.step(action)
		state = self.normalize(state)
		# store last results
		self.last_state = state
		self.last_action = action
		self.last_reward = reward
		# complete step
		self.cumulative_reward += reward
		self.step += 1
		if terminal: # add to statistics
			self.episodes.append( {"reward":self.cumulative_reward, "step":self.step} )
			if len(self.episodes) > flags.match_count_for_evaluation:
				self.episodes.popleft()
		return state, reward, terminal