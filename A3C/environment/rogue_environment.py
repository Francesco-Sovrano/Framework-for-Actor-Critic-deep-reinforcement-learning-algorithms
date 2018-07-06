# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# get command line args
import options
flags = options.get()

import math
import datetime
import logging
import sys
sys.path.append(flags.rogueinabox_path)
import numpy as np

from environment import environment
from rogueinabox.box import RogueBox

class RogueEnvironment(environment.Environment):
	def get_action_size(self):
		return len(self.real_actions)
		
	def get_state_shape(self):
		return self.game.state_generator._shape
	
	def __init__(self, thread_index):
		environment.Environment.__init__(self)
		self.thread_index = thread_index
		self.real_actions = RogueBox.get_actions()
		self.game = RogueBox(flags.env_path, flags.state_generator, flags.reward_generator, flags.steps_per_episode, flags.match_count_for_evaluation)

	def reset(self):
		self.last_action, status = self.game.reset()
		self.last_reward, new_state, _, _ = status
		self.last_state = new_state["value"]
		
	def stop(self):
		self.game.stop()
		
	def get_statistics(self):
		return self.game.evaluator.statistics()
		
	def get_screen_shape(self):
		return self.game.state_generator.screen_shape()
		
	def get_screen(self):
		return self.game.get_screen()
		
	def get_frame_info(self, value_estimator_network):
		# Screen
		last_frame = self.game.get_frame(-1)	
		screen_info = {
			"reward": self.last_reward,
			"passages": last_frame.get_tile_count("#"),
			"doors": last_frame.get_tile_count("+"),
			"below_player": last_frame.get_tile_below_player(),
		}
		augmented_screen = [str(["{0}={1}".format(key,value) for key, value in screen_info.items()]) + '\n'] + self.get_screen()
		frame_dict = { "screen": '\n'.join(augmented_screen) }
		# Heatmap
		if flags.save_episode_heatmap:
			heatmap_states = self.game.compute_walkable_states()
			(screen_x,screen_y,_) = self.get_screen_shape()
			value_map = np.zeros((screen_x, screen_y))
			concat=self.get_last_action_reward()
			for (heatmap_state,(x,y)) in heatmap_states:
				value_map[x][y] = value_estimator_network.estimate_value(state=heatmap_state, concat=concat)
			frame_dict.update( { "heatmap": value_map } )
		# return
		return frame_dict

	def get_last_action_reward(self):
		action_reward = np.zeros(len(self.real_actions)+1, dtype=np.uint8)
		action_reward[self.last_action]=1
		action_reward[-1] = self.last_reward
		return action_reward
		
	def process(self, action):
		action = action%len(self.real_actions)
		real_action = self.real_actions[action]
		reward, new_state, win, lose = self.game.send_command(real_action)
		new_state = new_state["value"]
		
		self.last_state = new_state
		self.last_action = action
		self.last_reward = reward
		return new_state, reward, (win or lose)