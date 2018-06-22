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
		self.level = 0
		self.tiles = 0
		for t in range(len(infos)):
			info = infos[t]
			if not info.statusbar["is_empty"]:
				current_level = info.statusbar["dungeon_level"]-1
				if self.level < current_level:
					self.level = current_level
					self.tiles += infos[t-1].get_known_tiles_count()
		if len(infos) > 1:
			if infos[-1].statusbar["dungeon_level"] == infos[-2].statusbar["dungeon_level"]:
				self.tiles += infos[-1].get_known_tiles_count()
		self.reward = reward
		self.has_won = has_won
		self.step = step
	
class RogueEvaluator:

	def __init__(self, match_count_for_evaluation):
		self.reset()
		self.match_count_for_evaluation = match_count_for_evaluation
		
	def reset(self):
		self.episodes = collections.deque()
		self.min_reward = 0
		self.max_reward = 0
		
	def add(self, info, reward, has_won, step): # O(1)
		self.episodes.append( Episode(info, reward, has_won, step) )
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
			for e in self.episodes:
				result["avg_steps"] += e.step
				result["avg_reward"] += e.reward
				result["avg_tiles"] += e.tiles
				result["avg_level"] += e.level
				# if e.has_won:
					# result["avg_success_steps"] += e.step
					# victories +=1
			result["avg_steps"] /= count
			result["avg_reward"] /= count
			result["avg_tiles"] /= count
			result["avg_level"] /= count
			# if victories > 0:
				# result["avg_success_steps"] /= victories
		return result
		