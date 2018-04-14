#Copyright (C) 2017 Andrea Asperti, Carlo De Pieri, Gianmaria Pedrini, Francesco Sovrano
#
#This file is part of Rogueinabox.
#
#Rogueinabox is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#Rogueinabox is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

import itertools
import numpy as np
from abc import ABC, abstractmethod

# ABSTRACT CLASS

class RewardGenerator(ABC):
	def __init__(self):
		self.reset()
		self.set_default_reward()
	
	def reset(self):
		self.goal_achieved = False
		
	def set_default_reward(self):
		self.default_reward = 0

	def compute_reward(self, frame_history):
		if len(frame_history) < 2:
			return self.default_reward
		new_info = frame_history[-1]
		old_info = frame_history[-2]
		if old_info.has_statusbar() and new_info.has_statusbar():
			return self.transform_value( self.get_value(old_info, new_info) )
		return self.default_reward
	
	def get_value (self, old_info, new_info):
		return 0
		
	def transform_value (self, reward):
		return reward
		
	@staticmethod
	def manhattan_distance(a, b):
		return abs(a[0] - b[0]) + abs(a[1] - b[1])
		
	@staticmethod
	def player_standing_still(old_info, new_info):
		return old_info.get_player_pos() == new_info.get_player_pos()
		
	@staticmethod
	def remap( x, oMax, nMax ):
		#range check
		oMin = -oMax
		nMin = -nMax

		#check reversed input range
		reverseInput = False
		oldMin = min( oMin, oMax )
		oldMax = max( oMin, oMax )
		if not oldMin == oMin:
			reverseInput = True

		#check reversed output range
		reverseOutput = False   
		newMin = min( nMin, nMax )
		newMax = max( nMin, nMax )
		if not newMin == nMin :
			reverseOutput = True

		portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
		if reverseInput:
			portion = (oldMax-x)*(newMax-newMin)/(oldMax-oldMin)

		result = portion + newMin
		if reverseOutput:
			result = newMax - portion

		return result
		
class E_D_W_RewardGenerator(RewardGenerator):
	def set_default_reward(self):
		self.default_reward = -1
		
	def get_value (self, old_info, new_info):
		"""return the reward for the last action
		+100 for descending the stairs
		+5 for exploring the map
		-0.1 living reward
		"""
		if new_info.statusbar["dungeon_level"] > old_info.statusbar["dungeon_level"]:
			self.goal_achieved = True
			return 100
		elif new_info.get_known_tiles_count() > old_info.get_known_tiles_count():
			return 5
		else:
			return -0.1
		
class E_D_Ps_W_RewardGenerator(E_D_W_RewardGenerator):
	def get_value (self, old_info, new_info):
		"""return the reward the last action
		+100 for descending the stairs
		+5 for exploring the map
		-1 for standing still
		-0.1 living reward
		"""
		if self.player_standing_still(old_info, new_info):
			return -1
		return super().get_value(old_info, new_info)
						
class Clipped_RewardGenerator(RewardGenerator):
	def transform_value (self, reward):
		return np.clip(reward, -1, 1)
		
	def get_value (self, old_info, new_info):
		if new_info.statusbar["dungeon_level"] > old_info.statusbar["dungeon_level"]:
			self.goal_achieved = True
			return 10000
		elif new_info.get_tile_count("+") > old_info.get_tile_count("+"): # doors
			return 100
		elif new_info.get_tile_count("#") > old_info.get_tile_count("#"): # passages
			return 1
		elif self.player_standing_still(old_info, new_info): #standing reward
			return -0.05
		return 0
		
class Normalised_RewardGenerator(RewardGenerator):
	def transform_value (self, reward):
		return self.remap( reward, 500, 1 ) # from [-500,500] to [-1,1]
		
	def get_value (self, old_info, new_info):
		if new_info.statusbar["dungeon_level"] > old_info.statusbar["dungeon_level"]:
			self.goal_achieved = True
			return 250
		elif new_info.get_tile_count("+") > old_info.get_tile_count("+"): # doors
			return 10
		elif new_info.get_tile_count("#") > old_info.get_tile_count("#"): # passages
			return 1
		elif self.player_standing_still(old_info, new_info): #standing reward
			return -1
		return 0
		
class StairSeeker_RewardGenerator(RewardGenerator):

	def get_value (self, old_info, new_info):
		if new_info.statusbar["dungeon_level"] > old_info.statusbar["dungeon_level"]:
			self.goal_achieved = True
			return 10
		elif new_info.get_tile_count("+") > old_info.get_tile_count("+"): # doors
			return 1
		elif self.player_standing_still(old_info, new_info): #standing reward
			return -0.01
		return 0
		
class ImprovedStairSeeker_RewardGenerator(StairSeeker_RewardGenerator):

	def get_value (self, old_info, new_info):
		if old_info.get_tile_below_player() == '+' and new_info.get_tile_count("#") > old_info.get_tile_count("#"): # has opened a new door
			return 1
		return super().get_value(old_info, new_info)
		
class ImprovedStairSeeker_2_RewardGenerator(StairSeeker_RewardGenerator):

	def get_value (self, old_info, new_info):
		if old_info.get_tile_below_player() == '+' and new_info.get_tile_count("#") > old_info.get_tile_count("#"): # has opened a new door
			return 0.5
		return super().get_value(old_info, new_info)
		
class ImprovedStairSeeker_3_RewardGenerator(StairSeeker_RewardGenerator):

	def get_value (self, old_info, new_info):
		if old_info.get_tile_below_player() == '+' and new_info.get_tile_count("#") > old_info.get_tile_count("#"): # has opened a new door
			return 0.25
		return super().get_value(old_info, new_info)
		
class BalancedStairSeeker_RewardGenerator(RewardGenerator):
	
	def get_value (self, old_info, new_info):
		if new_info.statusbar["dungeon_level"] > old_info.statusbar["dungeon_level"]:
			self.goal_achieved = True
			return 10
		if old_info.get_tile_below_player() == '+' and new_info.get_tile_count("#") > old_info.get_tile_count("#"): # has opened a new door
			return 0.125
		elif new_info.get_tile_count("+") > old_info.get_tile_count("+"): # doors
			return 0.5
		elif self.player_standing_still(old_info, new_info): #standing reward
			return -0.01
		return 0
		
class SuperStairSeeker_RewardGenerator(RewardGenerator):
	
	def get_value (self, old_info, new_info):
		if new_info.statusbar["dungeon_level"] > old_info.statusbar["dungeon_level"]:
			self.goal_achieved = True
			return 100
		elif old_info.get_tile_below_player() == '+' and new_info.get_tile_count("#") > old_info.get_tile_count("#"): # has opened a new door
			return 1
		elif new_info.get_tile_count("+") > old_info.get_tile_count("+"): # doors
			return 1
		elif self.player_standing_still(old_info, new_info): #standing reward
			return -0.01
		return 0