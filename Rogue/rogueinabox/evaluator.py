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

# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

class Episode:
	def __init__(self, infos, reward, has_won, step):
		self.reward = reward
		self.has_won = has_won
		self.step = step
		self.level = 0
		self.tiles = 0
		# count level and tiles
		infos_length = len(infos)
		if infos_length < 1:
			return
		for t in range(infos_length):
			info = infos[t]
			if not info.statusbar["is_empty"]: # the statusbar is not empty
				current_level = info.statusbar["dungeon_level"]-1
				if self.level < current_level: # level has changed
					self.level = current_level
					self.tiles += infos[t-1].get_known_tiles_count() # is the sum of all the tiles discovered in every level
		if infos_length == 1 or infos[-1].statusbar["dungeon_level"] == infos[-2].statusbar["dungeon_level"]: # the last frame was not a level change
			self.tiles += infos[-1].get_known_tiles_count() # add the current amount of tiles of the current level
	
class RogueEvaluator:

	def __init__(self, match_count_for_evaluation):
		self.reset()
		self.match_count_for_evaluation = match_count_for_evaluation
		
	def reset(self):
		self.episodes = collections.deque()
		self.min_reward = 0
		self.max_reward = 0
		
	def add(self, infos, reward, has_won, step): # O(1)
		self.episodes.append( Episode(infos, reward, has_won, step) )
		if len(self.episodes) > self.match_count_for_evaluation:
			self.episodes.popleft()
	
	def statistics(self): # O(self.match_count_for_evaluation)
		result = {}
		result["avg_reward"] = 0
		result["avg_tiles"] = 0
		result["avg_level"] = 0
		result["avg_steps"] = 0
		# result["avg_success_steps"] = 0
		
		count = len(self.episodes)
		# victories = 0
		if count>0:
			result["avg_steps"] = sum(e.step for e in self.episodes)/count
			result["avg_reward"] = sum(e.reward for e in self.episodes)/count
			result["avg_tiles"] = sum(e.tiles for e in self.episodes)/count
			result["avg_level"] = sum(e.level for e in self.episodes)/count
		return result
		