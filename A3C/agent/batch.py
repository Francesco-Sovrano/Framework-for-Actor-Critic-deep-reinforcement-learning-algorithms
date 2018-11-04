import numpy as np

def is_tuple(val):
	return type(val) in [list,tuple]

class ExperienceBatch(object):

	def __init__(self, model_size):
		self.model_size = model_size
		# action info
		self.states = [[] for _ in range(model_size)] # do NOT use [[]]*model_size
		self.concats = [[] for _ in range(model_size)]
		self.actions = [[] for _ in range(model_size)]
		self.policies = [[] for _ in range(model_size)]
		self.rewards = [[] for _ in range(model_size)]
		self.values = [[] for _ in range(model_size)]
		self.internal_states = [[] for _ in range(model_size)]
		# cumulative info
		self.discounted_cumulative_rewards = [None]*model_size
		self.generalized_advantage_estimators = [None]*model_size
		
		self.bootstrap = {}
		self.agent_position_list = []
		
	def reset_internal_states(self):
		self.internal_states = [[None] for _ in range(self.model_size)]
	
	def get_action(self, action, agent, pos):
		if not is_tuple(action):
			return self.__dict__[action][agent][pos]
		return (self.__dict__[key][agent][pos] for key in action)
		
	def set_action(self, feed_dict, agent, pos):
		for (key, value) in feed_dict.items():
			q = self.__dict__[key][agent]
			if len(q) <= pos: # add missing steps
				q.extend([None]*(pos-len(q)+1))
			q[pos] = value

	def add_action(self, agent_id, state, concat, action, policy, reward, value, internal_state=None):
		self.states[agent_id].append(state)
		self.concats[agent_id].append(concat)
		self.internal_states[agent_id].append(internal_state)
		self.rewards[agent_id].append(reward) # extrinsic + intrinsic reward
		self.values[agent_id].append(value)
		self.actions[agent_id].append(action)
		self.policies[agent_id].append(policy)
		
		self.agent_position_list.append( (agent_id, len(self.states[agent_id])-1) ) # (agent_id, batch_position)
		
	def get_cumulative_reward(self, agents=None):
		if agents is None:
			return sum( sum(rewards) for rewards in self.rewards )
		return sum( sum(rewards) for agent,rewards in enumerate(self.rewards) if agent in agents )
		
	def get_size(self, agents=None):
		if agents is None:
			return sum(len(s) for s in self.states)
		return sum(len(s) for agent,s in enumerate(self.states) if agent in agents)

	def step_generator(self, agents=None):
		if agents is None:
			return self.agent_position_list
		if len(agents)==1:
			agent = list(agents)[0]
			return ((agent,pos) for pos in range(self.get_size(agents)))
		return ((agent,pos) for (agent,pos) in self.agent_position_list if agent in agents)
		
	def reversed_step_generator(self, agents=None):
		if agents is None:
			return reversed(self.agent_position_list)
		if len(agents)==1:
			agent = list(agents)[0]
			return ((agent,pos) for pos in range(self.get_size(agents)-1,-1,-1))
		return ((agent,pos) for (agent,pos) in reversed(self.agent_position_list) if agent in agents)
		
	def compute_discounted_cumulative_reward(self, agents, last_value, gamma, lambd):
		# prepare batch
		for i in agents:
			self.discounted_cumulative_rewards[i]=[]
			self.generalized_advantage_estimators[i]=[]
		# bootstrap
		discounted_cumulative_reward = last_value
		generalized_advantage_estimator = 0.0
		# compute cumulative reward and advantage
		for (agent,pos) in self.reversed_step_generator(agents):
			reward, value = self.get_action(['rewards','values'], agent, pos)
			reward = np.sum(reward) # extrinsic + intrinsic reward
			discounted_cumulative_reward = reward + gamma*discounted_cumulative_reward
			generalized_advantage_estimator = reward + gamma*last_value - value + gamma*lambd*generalized_advantage_estimator
			feed_dict = {'discounted_cumulative_rewards':discounted_cumulative_reward, 'generalized_advantage_estimators':generalized_advantage_estimator}
			self.set_action(feed_dict, agent, pos)
			# update last value
			last_value = value
			
	def append(self, batch):
		pos_base = [len(self.states[agent]) for agent in range(self.model_size)]
		self.agent_position_list.extend((agent, pos+pos_base[agent]) for (agent,pos) in batch.agent_position_list)
		for i in range(self.model_size):
			self.states[i].extend(batch.states[i])
			self.concats[i].extend(batch.concats[i])
			self.actions[i].extend(batch.actions[i])
			self.policies[i].extend(batch.policies[i])
			self.rewards[i].extend(batch.rewards[i])
			self.values[i].extend(batch.values[i])
			self.internal_states[i].extend(batch.internal_states[i])
		self.bootstrap = batch.bootstrap