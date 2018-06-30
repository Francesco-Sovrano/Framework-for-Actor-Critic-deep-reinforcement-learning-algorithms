# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import traceback
import collections
import os
import logging
import numpy as np
import time

from environment.environment import Environment
from model.model_manager import ModelManager

# get command line args
import options
flags = options.get()

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000


class Worker(object):
	def __init__(self, thread_index, session, global_network, device, train=True):
		self.train = train
		self.thread_index = thread_index
		self.global_network = global_network
		#logs
		if not os.path.isdir(flags.log_dir + "/performance"):
			os.mkdir(flags.log_dir + "/performance")
		formatter = logging.Formatter('%(asctime)s %(message)s')
		# reward logger
		self.reward_logger = logging.getLogger('reward_' + str(thread_index))
		hdlr = logging.FileHandler(flags.log_dir + '/performance/reward_' + str(thread_index) + '.log')
		hdlr.setFormatter(formatter)
		self.reward_logger.addHandler(hdlr) 
		self.reward_logger.setLevel(logging.DEBUG)
		self.max_reward = float("-inf")
		# build network
		self.environment = Environment.create_environment(flags.env_type, self.thread_index)
		self.device = device
		if self.train:
			state_shape = self.environment.get_state_shape()
			action_size = self.environment.get_action_size()
			concat_size = action_size+1
			self.local_network = ModelManager(session=session, device=self.device, id=self.thread_index, action_size=action_size, concat_size=concat_size, state_shape=state_shape, global_network=self.global_network)
		else:
			self.local_network = self.global_network
		self.terminal = True
		self.local_t = 0
		self.prev_local_t = 0
		self.terminated_episodes = 0
		self.stats = {}

	def update_statistics(self):
		self.stats = self.environment.get_statistics()
		self.stats.update(self.local_network.get_statistics())

	def prepare(self): # initialize a new episode
		self.terminal = False
		self.episode_reward = 0
		self.environment.reset()
		self.local_network.reset()

	def stop(self): # stop current episode
		self.environment.stop()
		
	def set_start_time(self, start_time):
		self.start_time = start_time

	def _print_log(self, global_t, step):
		self.local_t += step
		if (self.thread_index == 1) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
			self.prev_local_t += PERFORMANCE_LOG_INTERVAL
			elapsed_time = time.time() - self.start_time
			steps_per_sec = global_t / elapsed_time
			print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format( global_t, elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))
		# print statistics
		if self.terminal:
			self._print_statistics(global_t)
		
	def _print_statistics(self, global_t):
		# Update statistics
		self.update_statistics()
		# Print statistics
		self.reward_logger.info( str(["{0}={1}".format(key,value) for key, value in self.stats.items()]) )
		# show episodes insides
		if flags.show_all_screenshots:
			self.environment.print_display(global_t, self.episode_reward)
		elif flags.show_best_screenshots:
			if self.episode_reward >= self.max_reward:
				self.max_reward = self.episode_reward
				self.environment.print_display(global_t, self.episode_reward)
				
	# run simulations
	def run_batch(self):
		if self.train: # Copy weights from shared to local
			self.local_network.sync_with_global()			
		self.local_network.reset_batch()
			
		step = 0
		while step < flags.max_batch_size and not self.terminal:
			policy, value = self.local_network.compute_policy_value(state=self.environment.last_state, concat=self.environment.get_last_action_reward())
			action = self.environment.choose_action(policy)
			_, reward, self.terminal = self.environment.process(action)
			self.local_network.save_action_reward(action, reward)
			
			self.episode_reward += reward
			step += 1
		if self.terminal: # an episode has terminated
			self.terminated_episodes += 1
			
		if self.train: # train using batch
			if self.terminal: # no bootstrap
				self.local_network.save_batch()
			else: # bootstrap
				self.local_network.save_batch(state=self.environment.last_state, concat=self.environment.get_last_action_reward())
		return step

	def process(self, global_t = 0):
		try:
			if self.terminal:
				self.prepare()
			step = self.run_batch()
			# print log
			self._print_log(global_t, step)
			return step
		except:
			traceback.print_exc()
			self.terminal = True
			return 0
			
	def build_value_map(self):
		self.prepare()
		while not self.terminal:
			self.process()
		(screen_x,screen_y,_) = self.environment.get_screen_shape()
		value_map = np.zeros((screen_x, screen_y))
		walkable_states = self.environment.compute_heatmap_states()
		for (state,(x,y)) in walkable_states:
			_, value = self.local_network.compute_policy_value(state=state, concat=self.environment.get_last_action_reward())
			value_map[x][y] = value
		return value_map