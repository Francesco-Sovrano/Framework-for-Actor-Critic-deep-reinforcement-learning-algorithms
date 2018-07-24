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
		(self.last_action, (self.last_reward, new_state, _, _)) = self.game.reset()
		self.last_state = new_state["value"]
		
	def stop(self):
		self.game.stop()
		
	def get_statistics(self):
		return self.game.evaluator.statistics()
		
	def get_screen_shape(self):
		return self.game.state_generator.screen_shape()
		
	def get_screen(self):
		return self.game.get_screen()
		
	def get_frame_info(self, network, observation, policy, value, action, reward):
		# Screen
		last_frame = self.game.get_frame(-2)	
		state_info = "reward={}, passages={}, doors={}, below_player={}, agent={}, action={}, value={}\n".format( 
			reward,
			last_frame.get_tile_count("#"),
			last_frame.get_tile_count("+"),
			last_frame.get_tile_below_player(),
			network.agent_id,
			action,
			value
		)
		policy_info = "policy={}\n".format(policy)
		# observation_info = "observation={}".format(np.array_str(observation.flatten()))
		frame_dict = {}
		frame_dict["log"] = state_info + policy_info + '\n'.join(last_frame.screen)+'\n'
		frame_dict["screen"] = { "value": frame_dict["log"], "type": 'ASCII' }
		# Heatmap
		if flags.save_episode_heatmap:
			heatmap_states = self.game.compute_walkable_states()
			(screen_x,screen_y,_) = self.get_screen_shape()
			value_map = np.zeros((screen_x, screen_y))
			concat=self.get_last_action_reward()
			for (heatmap_state,(x,y)) in heatmap_states:
				value_map[x][y] = network.estimate_value(state=heatmap_state, concat=concat)
			frame_dict["heatmap"] = value_map
			
		return frame_dict
		
	def process(self, action):
		action = action%len(self.real_actions)
		real_action = self.real_actions[action]
		reward, new_state, win, lose = self.game.send_command(real_action)
		
		self.last_state = new_state["value"]
		self.last_action = action
		self.last_reward = reward
		return self.last_state, reward, (win or lose)