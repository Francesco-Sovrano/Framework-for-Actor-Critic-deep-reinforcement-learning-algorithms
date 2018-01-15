# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Episode:
	def __init__(self, info, reward, has_won, step):
		self.info = info
		self.reward = reward
		self.has_won = has_won
		self.step = step
	
class RogueEvaluator:

	def __init__(self):
		self.reset()
		
	def reset(self):
		self.episodes = []
		self.min_reward = 0
		self.max_reward = 0
		
	def add(self, info, reward, has_won, step):
		self.episodes.append( Episode(info, reward, has_won, step) )
	
	def statistics(self, count = 10):
		result = {}
		result["win_perc"] = 0
		result["reward_avg"] = 0
		result["tiles_avg"] = 0
		result["steps_avg"] = 0
		if count <= 0:
			count = 1
		if count > len(self.episodes):
			count = len(self.episodes)
		
		for e in self.episodes[-int(count):]:
			if e.has_won:
				result["win_perc"] += 1
				result["steps_avg"] += e.step
			result["reward_avg"] += e.reward
			result["tiles_avg"] += e.info.get_known_tiles_count()
		result["win_perc"] /= count
		result["reward_avg"] /= count
		result["tiles_avg"] /= count
		result["steps_avg"] /= count
		return result
		