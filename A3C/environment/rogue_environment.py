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
		if flags.show_best_screenshots or flags.show_all_screenshots:
			self.screenshots = list()
			self.screeninfo = list()
		self.last_action, status = self.game.reset()
		self.last_reward, new_state, _, _ = status
		self.last_state = new_state["value"]
		
	def stop(self):
		self.game.stop()
		
	def get_screen(self):
		return self.game.get_screen()
		
	def get_statistics(self):
		return self.game.evaluator.statistics()
		
	def _save_display(self):
		self.screenshots.append( self.get_screen() )
		last_frame = self.game.frame_info[-1]
		self.screeninfo.append( "reward: {2}, passages: {0}, doors: {1}, below_player: {3}\n".format(
			last_frame.get_tile_count("#"),
			last_frame.get_tile_count("+"), 
			self.game.reward,
			last_frame.get_tile_below_player() )
		)
		
	def print_display(self, step, reward):
		file = open(flags.log_dir + '/screenshots/reward(' + str(reward) + ')_step(' + str(step) + ')_thread(' + str(self.thread_index) + ').log',"w") 
		for i in range(len(self.screenshots)):
			file.write( self.screeninfo[i] )
			screen = self.screenshots[i]
			for line in screen:
				file.write( str(line) + '\n' )
		file.close()
		
	def process(self, action):
		action = action%len(self.real_actions)
		real_action = self.real_actions[action]
		reward, new_state, win, lose = self.game.send_command(real_action)
		new_state = new_state["value"]
		
		self.last_state = new_state
		self.last_action = action
		self.last_reward = reward
		if flags.show_best_screenshots or flags.show_all_screenshots:
			self._save_display()
		return new_state, reward, (win or lose)

	def get_last_action_reward(self):
		action_reward = np.zeros(len(self.real_actions)+1, dtype=np.uint8)
		action_reward[self.last_action]=1
		action_reward[-1] = self.last_reward
		return action_reward