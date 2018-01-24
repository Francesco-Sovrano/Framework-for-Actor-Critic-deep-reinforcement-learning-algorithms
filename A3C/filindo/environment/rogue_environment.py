# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# get command line args
import options
flags = options.get()

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
		
	def get_layers_count(self):
		return self.game.state_generator._situations
	
	def __init__(self, thread_index):
		environment.Environment.__init__(self)
		self.thread_index = thread_index
		self.real_actions = RogueBox.get_actions()
		self.game = RogueBox(flags.env_path, flags.state_generator, flags.reward_generator, flags.steps_per_episode)

	def reset(self):
		if flags.show_best_screenshots or flags.show_all_screenshots:
			self.screenshots = list()
			self.screeninfo = list()
		self.last_action, status = self.game.reset()
		self.last_reward, self.last_state, _ = status
		
	def stop(self):
		self.game.stop()

	def _process_frame(self, action):
		return self.game.send_command(action)
		
	def get_screen(self):
		return self.game.get_screen()
		
	def _save_display(self):
		self.screenshots.append( self.game.get_screen() )
		last_frame = self.game.frame_info[-1]
		self.screeninfo.append( "reward: {2}, passages: {0}, doors: {1}, below_player: ".format(
			last_frame.get_tile_count("#"),
			last_frame.get_tile_count("+"), 
			self.game.reward
		) + last_frame.get_tile_below_player() + "\n" )
		
	def print_display(self, step, reward):
		file = open(flags.log_dir + '/screenshots/reward(' + str(reward) + ')_step(' + str(step) + ')_thread(' + str(self.thread_index) + ').log',"w") 
		for i in range(len(self.screenshots)):
			file.write( self.screeninfo[i] )
			screen = self.screenshots[i]
			for line in screen:
				file.write( str(line) + '\n' )
		file.close() 
		
	def process(self, action):
		real_action = self.real_actions[action]
		reward, new_state, terminal = self._process_frame(real_action)
		
		self.last_state = new_state
		self.last_action = action
		self.last_reward = reward
		if flags.show_best_screenshots or flags.show_all_screenshots:
			self._save_display()
		return new_state, reward, terminal
