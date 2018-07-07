import numpy as np

class Buffer(object):
	def __init__(self, size):
		self.size = size
		self.clean()
		
	def clean(self):
		# Memory
		self.batches = [None]*self.size
		# Size indexes
		self.next_idx = 0
		self.num_in_buffer = 0

	def has_atleast(self, frames):
		return self.num_in_buffer >= frames
		
	def has(self, frames):
		return self.num_in_buffer == frames
		
	def is_full(self):
		return self.has(self.size)

	def put(self, batch):
		self.batches[self.next_idx] = batch
		self.next_idx = (self.next_idx + 1) % self.size
		self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

	def get(self):
		assert self.has_atleast(frames=1)
		idx = np.random.randint(0, self.num_in_buffer)
		return self.batches[idx]
