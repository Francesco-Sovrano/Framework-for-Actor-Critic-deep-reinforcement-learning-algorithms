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
		return new_info.statusbar["dungeon_level"] == old_info.statusbar["dungeon_level"] and old_info.get_player_pos() == new_info.get_player_pos()
		
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
		elif new_info.get_tile_count("+") > old_info.get_tile_count("+"): # new doors discovered
			return 100
		elif new_info.get_tile_count("#") > old_info.get_tile_count("#"): # new passages discovered
			return 1
		elif self.player_standing_still(old_info, new_info): # malus for not moving
			return -0.05
		return 0
		
class Normalised_RewardGenerator(RewardGenerator):
	def transform_value (self, reward):
		return self.remap( reward, 500, 1 ) # from [-500,500] to [-1,1]
		
	def get_value (self, old_info, new_info):
		if new_info.statusbar["dungeon_level"] > old_info.statusbar["dungeon_level"]:
			self.goal_achieved = True
			return 250
		elif new_info.get_tile_count("+") > old_info.get_tile_count("+"): # new doors discovered
			return 10
		elif new_info.get_tile_count("#") > old_info.get_tile_count("#"): # new passages discovered
			return 1
		elif self.player_standing_still(old_info, new_info): # malus for not moving
			return -1
		return 0
		
class Gold_RewardGenerator(RewardGenerator):
	def get_value (self, old_info, new_info):
		if new_info.statusbar["gold"] > old_info.statusbar["gold"]:
			return new_info.statusbar["gold"]-old_info.statusbar["gold"]
		return 0
		
class Stair_RewardGenerator(RewardGenerator):
	def get_value (self, old_info, new_info):
		if new_info.statusbar["dungeon_level"] > old_info.statusbar["dungeon_level"]:
			self.goal_achieved = True
			return 10
		return 0
		
class NoStanding_S_RewardGenerator(Stair_RewardGenerator):
	def get_value (self, old_info, new_info):
		sup = super().get_value(old_info, new_info)
		if sup != 0:
			return sup
		if self.player_standing_still(old_info, new_info): # malus for not moving
			return -0.01
		return 0
		
class Explore_NSS_RewardGenerator(NoStanding_S_RewardGenerator):
	def get_value (self, old_info, new_info):
		sup = super().get_value(old_info, new_info)
		if sup != 0:
			return sup
		if new_info.get_tile_count("+") > old_info.get_tile_count("+"): # new doors discovered
			return 1
		return 0
		
class Improved_ENSS_RewardGenerator(Explore_NSS_RewardGenerator):
	def get_value (self, old_info, new_info):
		sup = super().get_value(old_info, new_info)
		if sup != 0:
			return sup
		if new_info.get_tile_below_player() == '+' and new_info.get_tile_count("#") > old_info.get_tile_count("#"): # has opened a new door
			return 1
		return 0
				
class Balanced_NSS_RewardGenerator(NoStanding_S_RewardGenerator):
	def get_value (self, old_info, new_info):
		sup = super().get_value(old_info, new_info)
		if sup != 0:
			return sup
		if new_info.get_tile_below_player() == '+' and new_info.get_tile_count("#") > old_info.get_tile_count("#"): # has opened a new door
			return 0.125
		elif new_info.get_tile_count("+") > old_info.get_tile_count("+"): # new doors discovered -> this reward is given twice per room
			return 0.5
		return 0
		
class Health_ISS_RewardGenerator(Improved_ENSS_RewardGenerator):
	def get_value (self, old_info, new_info):
		sup = super().get_value(old_info, new_info)
		if sup != 0:
			return sup
		if new_info.statusbar["current_hp"] < old_info.statusbar["current_hp"]: # malus for losing life
			return -0.02
		if new_info.get_type_count("items") < old_info.get_type_count("items"): # bonus for picking an item
			return 0.1
		return 0
		
class Monster_IENSS_RewardGenerator(Improved_ENSS_RewardGenerator):
	def get_value (self, old_info, new_info):
		if new_info.statusbar["tot_exp"] > old_info.statusbar["tot_exp"] or new_info.statusbar["exp_level"] > old_info.statusbar["exp_level"]: # bonus for getting new experience
			return 0.1
		return super().get_value(old_info, new_info)