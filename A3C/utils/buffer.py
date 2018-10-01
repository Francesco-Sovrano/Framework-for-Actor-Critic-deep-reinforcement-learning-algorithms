import numpy as np
from collections import deque
from sortedcontainers import SortedDict

class Buffer(object):
	# __slots__ = ('types', 'size', 'batches')
	
	def __init__(self, size):
		self.size = size
		self.clean()
		
	def clean(self):
		self.types = {}
		self.batches = []
		
	def get_batches(self, type_id=None):
		if type_id is None:
			result = []
			for value in self.types.values():
				result += self.batches[value]
			return result
		return self.batches[self.get_type(type_id)]

	def has_atleast(self, frames, type=None):
		return self.count(type) >= frames
		
	def has(self, frames, type=None):
		return self.count(type) == frames
		
	def count(self, type=None):
		if type is None:
			if len(self.batches) == 0:
				return 0
			return sum(len(batch) for batch in self.batches)
		return len(self.batches[type])
		
	def id_is_full(self, type_id):
		return self.has(self.size, self.get_type(type_id))
		
	def is_full(self, type=None):
		if type is None:
			return self.has(self.size*len(self.types))
		return self.has(self.size, type)
		
	def is_empty(self, type=None):
		return not self.has_atleast(1, type)
		
	def get_type(self, type_id):
		self.add_type(type_id)
		return self.types[type_id]
		
	def add_type(self, type_id):
		if type_id in self.types:
			return
		self.types[type_id] = len(self.types)
		self.batches.append(deque())

	def put(self, batch, type_id=0): # put batch into buffer
		type = self.get_type(type_id)
		if self.is_full(type):
			self.batches[type].popleft()
		self.batches[type].append(batch)

	def sample(self):
		# assert self.has_atleast(frames=1)
		type = np.random.choice( [value for value in self.types.values() if not self.is_empty(value)] )
		id = np.random.randint(0, len(self.batches[type]))
		return self.batches[type][id]

class PrioritizedBuffer(Buffer):
	
	def clean(self):
		super().clean()
		self.prefixsum = []
		self.priorities = []
		
	def get_batches(self, type_id=None):
		if type_id is None:
			result = []
			for type in self.types.values():
				result += self.batches[type].values()
			return result
		return self.batches[self.get_type(type_id)].values()
		
	def add_type(self, type_id):
		if type_id in self.types:
			return
		self.types[type_id] = len(self.types)
		self.batches.append(SortedDict())
		self.prefixsum.append([])
		self.priorities.append({})
		
	def get_priority_from_unique(self, unique):
		return float(unique.split('#', 1)[0])
		
	def build_unique(self, priority, count):
		return '{:.5f}#{}'.format(priority,count) # new batch has higher unique priority than old ones with same shared priority
		
	def put(self, batch, priority, type_id=0): # O(log)
		type = self.get_type(type_id)
		if self.is_full(type):
			old_unique_batch_priority, _ = self.batches[type].popitem(index=0) # argument with lowest priority is always 0 because buffer is sorted by priority
			old_priority = self.get_priority_from_unique(old_unique_batch_priority)
			if old_priority in self.priorities[type] and self.priorities[type][old_priority] == 1: # remove from priority dictionary in order to prevent buffer overflow
				del self.priorities[type][old_priority]
		priority_count = self.priorities[type][priority] if priority in self.priorities[type] else 0
		priority_count = (priority_count % self.size) + 1 # modular counter to avoid overflow
		self.priorities[type][priority] = priority_count
		unique_batch_priority = self.build_unique(priority,priority_count)
		self.batches[type].update({unique_batch_priority: batch}) # O(log)
		self.prefixsum[type] = None # compute prefixsum only if needed, when sampling
		
	def keyed_sample(self): # O(n) after a new put, O(log) otherwise
		type_id = np.random.choice( [key for key,value in self.types.items() if not self.is_empty(value)] )
		type = self.get_type(type_id)
		if self.prefixsum[type] is None: # compute prefixsum
			self.prefixsum[type] = np.cumsum([self.get_priority_from_unique(k) for k in self.batches[type].keys()]) # O(n)
		mass = np.random.random() * self.prefixsum[type][-1]
		idx = np.searchsorted(self.prefixsum[type], mass) # O(log) # Find arg of leftmost item greater than or equal to x
		keys = self.batches[type].keys()
		if idx == len(keys): # this may happen when self.prefixsum[type] is negative
			idx = -1
		return self.batches[type][keys[idx]], idx, type_id
		
	def sample(self): # O(n) after a new put, O(log) otherwise
		return self.keyed_sample()[0]

	def update_priority(self, idx, priority, type_id=0): # O(log)
		type = self.get_type(type_id)
		_, batch = self.batches[type].popitem(index=idx) # argument with lowest priority is always 0 because buffer is sorted by priority
		self.put(batch, priority, type_id)