import numpy as np
# from collections import deque
from utils.segment_tree import SumSegmentTree, MinSegmentTree

class Buffer(object):
	# __slots__ = ('types', 'size', 'batches')
	
	def __init__(self, size):
		self.size = size
		self.clean()
		
	def clean(self):
		self.types = {}
		self.batches = []
		self.batches_next_idx = []
		
	def get_batches(self, type_id=None):
		if type_id is None:
			result = []
			for type in range(len(self.types)):
				result += self.batches[type]
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
		self.batches.append([])
		self.batches_next_idx.append(0)

	def put(self, batch, type_id=0): # put batch into buffer
		type = self.get_type(type_id)
		idx = self.batches_next_idx[type]
		if self.is_full(type):
			self.batches[type][idx] = batch
		else:
			self.batches[type].append(batch)
		self.batches_next_idx[type] = (idx + 1) % self.size
		return idx

	def sample(self):
		# assert self.has_atleast(frames=1)
		type = np.random.choice([type for type in range(len(self.types)) if not self.is_empty(type)])
		id = np.random.randint(0, len(self.batches[type]))
		return self.batches[type][id]

class PrioritizedBuffer(Buffer):
	
	def __init__(self, size):
		self._eps = 1e-6
		self._alpha = 0.6
		self.it_capacity = 1
		while self.it_capacity < size:
			self.it_capacity *= 2
		super().__init__(size)

	def clean(self):
		super().clean()
		self._it_sum = []
		# self._it_min = []
		
	def add_type(self, type_id):
		if type_id in self.types:
			return
		super().add_type(type_id)
		self._it_sum.append(SumSegmentTree(self.it_capacity))
		# self._it_min.append(MinSegmentTree(self.it_capacity))
		
	def put(self, batch, priority, type_id=0):
		idx = super().put(batch, type_id)
		self.update_priority(idx, priority, type_id)
		return idx
		
	def sample(self):
		type = np.random.choice([type for type in range(len(self.types)) if not self.is_empty(type)])
		mass = np.random.random() * self._it_sum[type].sum(0, self.count(type)-1)
		idx = self._it_sum[type].find_prefixsum_idx(mass)
		# weight = (self._it_sum[idx]/self._it_min.min()) ** (-beta) # importance weight
		# return self.batches[0][idx], idx, weight # multiply weight for advantage
		return self.batches[type][idx], idx

	def update_priority(self, idx, priority, type_id=0): # priority is advantage
		type = self.get_type(type_id)
		# assert 0 <= idx < self.count()
		priority = np.abs(priority) + self._eps
		# self._it_min[idx] = self._it_sum[idx] = priority ** self._alpha
		self._it_sum[type][idx] = priority ** self._alpha
		# self._max_priority = max(self._max_priority, priority)