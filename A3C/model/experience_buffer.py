import numpy as np
from collections import deque

class Buffer(object):
	def __init__(self, size, types=1):
		self.types = types
		self.buffer_size = size//types
		self.total_size = size*types
		self.clean()
		
	def clean(self):
		self.batches = []
		self.next_idx = []
		self.num_in_buffer = []
		for _ in range(self.types):
			self.batches.append([None]*self.buffer_size)
			self.next_idx.append(0)
			self.num_in_buffer.append(0)

	def has_atleast(self, frames, type=None):
		if type is None:
			return sum(self.num_in_buffer) >= frames
		return self.num_in_buffer[type] >= frames
		
	def has(self, frames, type=None):
		if type is None:
			return sum(self.num_in_buffer) == frames
		return self.num_in_buffer[type] == frames
		
	def is_full(self, type=None):
		if type is None:
			return self.has(self.total_size)
		return self.has(self.buffer_size, type)
		
	def is_empty(self, type=None):
		return not self.has_atleast(1, type)

	def put(self, batch, type=0):
		# put batch into buffer
		self.batches[type][self.next_idx[type]] = batch
		# update buffer size
		self.next_idx[type] = (self.next_idx[type] + 1) % self.buffer_size
		self.num_in_buffer[type] = min(self.buffer_size, self.num_in_buffer[type] + 1)

	def get(self):
		assert self.has_atleast(frames=1)
		type = np.random.choice([type for type in range(self.types) if not self.is_empty(type)])
		id = np.random.randint(0, self.num_in_buffer[type])
		return self.batches[type][id]