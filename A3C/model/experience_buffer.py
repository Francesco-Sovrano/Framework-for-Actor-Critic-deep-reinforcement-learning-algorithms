import numpy as np
from collections import deque

class Buffer(object):
	def __init__(self, size, type_count=1):
		self.types = {}
		self.type_count = type_count
		self.buffer_size = size//type_count
		self.total_size = size*type_count
		self.clean()
		
	def clean(self):
		self.batches = [[None]*self.buffer_size]*self.type_count
		self.next_idx = [0]*self.type_count
		self.num_in_buffer = [0]*self.type_count

	def has_atleast(self, frames, type=None):
		if type is None:
			return sum(self.num_in_buffer) >= frames
		return self.num_in_buffer[type] >= frames
		
	def has(self, frames, type=None):
		if type is None:
			return sum(self.num_in_buffer) == frames
		return self.num_in_buffer[type] == frames
		
	def id_is_full(self, type_id):
		return self.has(self.buffer_size, self.get_type(type_id))
		
	def is_full(self, type=None):
		if type is None:
			return self.has(self.total_size)
		return self.has(self.buffer_size, type)
		
	def is_empty(self, type=None):
		return not self.has_atleast(1, type)
		
	def get_type(self, type_id):
		if type_id not in self.types:
			self.types[type_id] = len(self.types)
		return self.types[type_id]

	def put(self, batch, type_id=0):
		type = self.get_type(type_id)
		# put batch into buffer
		self.batches[type][self.next_idx[type]] = batch
		# update buffer size
		self.next_idx[type] = (self.next_idx[type] + 1) % self.buffer_size
		self.num_in_buffer[type] = min(self.buffer_size, self.num_in_buffer[type] + 1)

	def get(self):
		# assert self.has_atleast(frames=1)
		type = np.random.choice([type for type in range(self.type_count) if not self.is_empty(type)])
		id = np.random.randint(0, self.num_in_buffer[type])
		return self.batches[type][id]