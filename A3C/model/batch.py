from collections import deque

class ExperienceBatch(object):
	__slots__ = (
		'states', 'concats', 'actions', 'rewards', 'values', 'policies', 'lstm_states', 'discounted_cumulative_rewards', 'generalized_advantage_estimators',
		'agent_position_list', 'total_reward', 'size', 'is_terminal',
		'bootstrap', 'manager_value_list'
	)
	
	def __init__(self, model_size):		
		self.states = [deque() for _ in range(model_size)] # do NOT use [deque]*model_size
		self.concats = [deque() for _ in range(model_size)]
		self.actions = [deque() for _ in range(model_size)]
		self.rewards = [deque() for _ in range(model_size)]
		self.values = [deque() for _ in range(model_size)]
		self.policies = [deque() for _ in range(model_size)]
		self.lstm_states = [deque() for _ in range(model_size)]
		self.discounted_cumulative_rewards = [None for _ in range(model_size)]
		self.generalized_advantage_estimators = [None for _ in range(model_size)]
		
		self.bootstrap = {}
		self.agent_position_list = []
		self.total_reward = 0
		self.size = 0
		self.is_terminal = False
		
	def get_agent_and_pos(self, index):
		batch_size = self.size
		if index >= batch_size or index < -batch_size:
			return None
		return self.agent_position_list[index]
		
class RewardPredictionBatch(object):
	__slots__ = ('states', 'target')
	
	def __init__(self, states, target):
		self.states = states
		self.target = [target]