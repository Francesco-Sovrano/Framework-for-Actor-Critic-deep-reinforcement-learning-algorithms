import numpy as np

import options
flags = options.get()

class Buffer(object):
	def __init__(self, size=2500):
		self.size = 2500
		# Memory
		self.batches = [None]*self.size
		# Size indexes
		self.next_idx = 0
		self.num_in_buffer = 0

	def has_atleast(self, frames):
		return self.num_in_buffer >= frames

	def put(self, batch):
		self.batches[self.next_idx] = batch
		self.next_idx = (self.next_idx + 1) % self.size
		self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

	def get(self):
		assert self.has_atleast(frames=1)
		idx = np.random.randint(0, self.num_in_buffer)
		return self.batches[idx]
