# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

class Episode:
	def __init__(self, info, reward, has_won, step):
		self.info = info
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
		result["accuracy"] = 0
		result["avg_reward"] = 0
		result["avg_tiles"] = 0
		result["avg_steps"] = 0
		
		count = len(self.episodes)
		if count>0:
			for e in self.episodes:
				if e.has_won:
					result["accuracy"] += 1
					result["avg_steps"] += e.step
				result["avg_reward"] += e.reward
				result["avg_tiles"] += e.info.get_known_tiles_count()
			result["accuracy"] /= count
			result["avg_reward"] /= count
			result["avg_tiles"] /= count
			result["avg_steps"] /= count
		return result
		