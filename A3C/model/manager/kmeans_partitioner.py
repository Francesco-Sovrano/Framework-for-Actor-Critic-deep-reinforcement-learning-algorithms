# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import traceback
import threading
import numpy as np
import copy

from model.network.actor_critic_network import ActorCriticNetwork
from model.experience_buffer import Buffer
from model.manager import BasicManager
from sklearn.cluster import KMeans

import options
flags = options.get()

class KMeansPartitioner(BasicManager):
	def set_model_size(self):
		self.model_size = flags.partition_count # manager output size
		if self.model_size < 2:
			self.model_size = 2
			
	def build_agents(self, state_shape, action_size, concat_size):
		# partitioner
		if self.is_global_network():
			self.buffer = Buffer(size=flags.partitioner_training_set_size)
			self.partitioner = KMeans(n_clusters=self.model_size)
		self.partitioner_trained = False
		# agents
		self.model_list = []
		for i in range(self.model_size):
			agent=ActorCriticNetwork(session=self.session, id="{0}_{1}".format(self.id, i), state_shape=state_shape, policy_size=action_size, entropy_beta=flags.entropy_beta, clip=self.clip[i], device=self.device, concat_size=concat_size)
			self.model_list.append(agent)
		# bind partition nets to training net
		if self.is_global_network():
			self.bind_to_training_net()
			self.lock = threading.Lock()
			
	def bind_to_training_net(self):
		self.sync_list = []
		training_net = self.get_model(0)
		for i in range(1,self.model_size):
			partition_net = self.get_model(i)
			self.sync_list.append(partition_net.bind_sync(training_net)) # for synching local network with global one
			
	def sync_with_training_net(self):
		for i in range(1,self.model_size):
			self.model_list[i].sync(self.sync_list[i-1])
		
	def get_state_partition(self, state):
		id = self.partitioner.predict([state.flatten()])[0]
		# print(self.id, " ", id)
		self.add_to_statistics(id)
		return id
		
	def query_partitioner(self, step):
		return self.partitioner_trained and step%flags.partitioner_granularity==0
		
	def act(self, policy_to_action_function, act_function, state, concat=None):
		if self.query_partitioner(self.step):
			self.agent_id = self.get_state_partition(state)
		# populate partitioner training set
		if not self.partitioner_trained and not self.is_global_network():
			self.global_network.populate_partitioner(state)
			self.partitioner_trained = self.global_network.partitioner_trained
			if self.partitioner_trained:
				self.partitioner = copy.deepcopy(self.global_network.partitioner)
		return super().act(policy_to_action_function, act_function, state, concat)
		
	def populate_partitioner(self, state):
		# assert self.is_global_network(), 'only global network can populate partitioner'
		with self.lock:
			self.buffer.put(batch=state.flatten())
			if self.buffer.is_full():
				print ("Buffer is full, starting partitioner training")
				self.partitioner.fit( self.buffer.batches )
				print ("Partitioner trained")
				self.partitioner_trained = True
				print ("Syncing with training net")
				self.sync_with_training_net()
				print ("Cleaning buffer")
				self.buffer.clean()
			
	def compute_cumulative_reward(self, state=None, concat=None):
		# Bootstrap partitioner
		bootstrap = state is not None
		if bootstrap and self.query_partitioner(self.step): # bootstrap the value from the last state
			self.agent_id = self.get_state_partition(state)
		# Compute agents' cumulative_reward
		super().compute_cumulative_reward(state, concat)