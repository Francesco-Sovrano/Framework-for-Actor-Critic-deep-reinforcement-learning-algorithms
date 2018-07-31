import sys
import numpy as np

class Buffer(object):
	def __init__(self, size):
		self.types = {}
		self.size = size
		self.clean()
		
	def clean(self):
		self.batches = []
		self.next_idx = []
		self.num_in_buffer = []

	def has_atleast(self, frames, type=None):
		if type is None:
			return sum(self.num_in_buffer) >= frames
		return self.num_in_buffer[type] >= frames
		
	def has(self, frames, type=None):
		if type is None:
			return sum(self.num_in_buffer) == frames
		return self.num_in_buffer[type] == frames
		
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
			self.batches.append([None]*self.size)
			self.next_idx.append(0)
			self.num_in_buffer.append(0)
		return self.types[type_id]

	def put(self, batch, type_id=0):
		type = self.get_type(type_id)
		# put batch into buffer
		self.batches[type][self.next_idx[type]] = batch
		# update buffer size
		self.next_idx[type] = (self.next_idx[type] + 1) % self.size
		self.num_in_buffer[type] = min(self.size, self.num_in_buffer[type] + 1)
		# if self.num_in_buffer[type] == self.size-1:
			# print("Buffer type ", type, " full with memory size ", sys.getsizeof(self))

	def get(self):
		# assert self.has_atleast(frames=1)
		type = np.random.choice([type for type in range(len(self.types)) if not self.is_empty(type)])
		id = np.random.randint(0, self.num_in_buffer[type])
		return self.batches[type][id]