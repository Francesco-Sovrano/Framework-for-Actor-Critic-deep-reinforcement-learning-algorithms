from collections import deque
import numpy as np

class ExperienceBatch(object):
	# __slots__ = ('bootstrap','agent_position_list','total_reward','size')

	def __init__(self, model_size):
		# action info
		self.states = [deque() for _ in range(model_size)] # do NOT use [deque]*model_size
		self.concats = [deque() for _ in range(model_size)]
		self.actions = [deque() for _ in range(model_size)]
		self.policies = [deque() for _ in range(model_size)]
		self.rewards = [deque() for _ in range(model_size)]
		self.values = [deque() for _ in range(model_size)]
		# cumulative info
		self.discounted_cumulative_rewards = [None for _ in range(model_size)]
		self.generalized_advantage_estimators = [None for _ in range(model_size)]
		
		self.bootstrap = {}
		self.agent_position_list = []
		self.total_reward = 0
		self.size = 0

	# def finalize(self):
		# for key in self.__dict__:
			# self.__dict__[key] = np.array(self.__dict__[key])
		# return self

	def get_step_action(self, action, step):
		id, pos = self.get_agent_and_pos(step)
		if type(action) not in [list,tuple]:
			return self.__dict__[action][id][pos]
		return (self.__dict__[key][id][pos] for key in action)

	def set_step_action(self, feed_dict, step):
		id, pos = self.get_agent_and_pos(step)
		for (key, value) in feed_dict.items():
			q = self.__dict__[key][id]
			if len(q) <= pos: # add missing steps
				q.extend([None]*(pos-len(q)+1))
			q[pos] = value
		
	def get_agent_and_pos(self, index):
		return self.agent_position_list[index]

	def add_agent_action(self, agent_id, state, concat, action, policy, reward, value, memorize_step=True):
		self.states[agent_id].append(state)
		self.concats[agent_id].append(concat)
		self.rewards[agent_id].append(reward)
		self.values[agent_id].append(value)
		self.actions[agent_id].append(action)
		self.policies[agent_id].append(policy)
		if memorize_step:
			self.agent_position_list.append( (agent_id, len(self.states[agent_id])-1) ) # (agent_id, batch_position)
			self.size += 1
			self.total_reward += reward
		
class RewardPredictionBatch(object):
	__slots__ = ('states', 'target')
	
	def __init__(self, states, target):
		self.states = states
		self.target = target