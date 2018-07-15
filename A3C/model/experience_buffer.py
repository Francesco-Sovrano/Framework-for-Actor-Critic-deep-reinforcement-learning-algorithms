import numpy as np
from collections import deque

class Buffer(object):
	def __init__(self, size):
		self.size = size
		self.clean()
		
	def clean(self):
		# Memory
		self.batches = [None]*self.size
		self.batch_ids_per_type = {}
		# Size indexes
		self.next_idx = 0
		self.num_in_buffer = 0

	def has_atleast(self, frames):
		return self.num_in_buffer >= frames
		
	def has(self, frames):
		return self.num_in_buffer == frames
		
	def is_full(self):
		return self.has(self.size)
		
	def is_empty(self):
		return not self.has_atleast(1)

	def put(self, batch, type):
		# remove old id
		if self.is_full():
			(_, type) = self.batches[self.next_idx]
			self.batch_ids_per_type[type].popleft()
		# add new id
		self.batches[self.next_idx] = (batch,type)
		if type not in self.batch_ids_per_type: # initialize type
			self.batch_ids_per_type[type] = deque()
		self.batch_ids_per_type[type].append(self.next_idx)
		# update buffer size
		self.next_idx = (self.next_idx + 1) % self.size
		self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

	def get(self):
		assert self.has_atleast(frames=1)
		type = np.random.choice([type for type in self.batch_ids_per_type if len(self.batch_ids_per_type[type])>0])
		id = np.random.choice(self.batch_ids_per_type[type])
		return self.batches[id]