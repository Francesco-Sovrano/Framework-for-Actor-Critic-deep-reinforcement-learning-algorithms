import sys
import numpy as np
from collections import deque

class Buffer(object):
	__slots__ = ('types', 'size', 'batches')
	
	def __init__(self, size):
		self.types = {}
		self.size = size
		self.clean()
		
	def clean(self):
		self.batches = []

	def has_atleast(self, frames, type=None):
		if type is None:
			if len(self.batches) == 0:
				return 0 >= frames
			return sum([len(batch) for batch in self.batches]) >= frames
		return len(self.batches[type]) >= frames
		
	def has(self, frames, type=None):
		if type is None:
			if len(self.batches) == 0:
				return 0 == frames
			return sum([len(batch) for batch in self.batches]) == frames
		return len(self.batches[type]) == frames
		
	def id_is_full(self, type_id):
		return self.has(self.size, self.get_type(type_id))
		
	def is_full(self, type=None):
		if type is None:
			return self.has(self.size*len(self.types))
		return self.has(self.size, type)
		
	def is_empty(self, type=None):
		return not self.has_atleast(1, type)
		
	def get_type(self, type_id):
		if type_id not in self.types:
			self.types[type_id] = len(self.types)
			self.batches.append(deque())
		return self.types[type_id]

	def put(self, batch, type_id=0):
		type = self.get_type(type_id)
		# put batch into buffer
		if self.is_full(type):
			self.batches[type].popleft()
		self.batches[type].append(batch)

	def get(self):
		# assert self.has_atleast(frames=1)
		type = np.random.choice([type for type in range(len(self.types)) if not self.is_empty(type)])
		id = np.random.randint(0, len(self.batches[type]))
		return self.batches[type][id]