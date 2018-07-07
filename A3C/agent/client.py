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

import agent.plots as plt

from environment.environment import Environment
from model.manager import *

# get command line args
import options
flags = options.get()

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000


class Worker(object):
	def get_model_manager(self):
		if flags.partition_count < 2:
			return "BasicManager"
		return flags.partitioner_type + "Partitioner"
			
	def __init__(self, thread_index, session, global_network, device, train=True):
		self.train = train
		self.thread_index = thread_index
		self.global_network = global_network
		#logs
		if not os.path.isdir(flags.log_dir + "/performance"):
			os.mkdir(flags.log_dir + "/performance")
		if not os.path.isdir(flags.log_dir + "/episodes"):
			os.mkdir(flags.log_dir + "/episodes")
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
			self.local_network = eval(self.get_model_manager())(session=session, device=self.device, id=self.thread_index, action_size=action_size, concat_size=concat_size, state_shape=state_shape, global_network=self.global_network)
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
		self.frame_info_list = []
		self.environment.reset()
		self.local_network.reset_LSTM()

	def stop(self): # stop current episode
		self.environment.stop()
		
	def set_start_time(self, start_time):
		self.start_time = start_time
		
	def print_speed(self, global_t, step):
		self.local_t += step
		if (self.thread_index == 1) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
			self.prev_local_t += PERFORMANCE_LOG_INTERVAL
			elapsed_time = time.time() - self.start_time
			steps_per_sec = global_t / elapsed_time
			print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format( global_t, elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))
		
	def print_statistics(self):
		# Update statistics
		self.update_statistics()
		# Print statistics
		self.reward_logger.info( str(["{0}={1}".format(key,value) for key, value in self.stats.items()]) )
		
	def print_frames(self, global_step):
		frames_count = len(self.frame_info_list)
		if frames_count == 0:
			return
		episode_directory = "{0}/episodes/reward({1})_step({2})_thread({3})".format(flags.log_dir, self.episode_reward, global_step, self.thread_index)
		os.mkdir(episode_directory)
		# Screen
		os.mkdir(episode_directory+'/screens')
		screen_filenames = []
		with open(episode_directory + '/screen.log',"w") as screen_file:
			for i in range(frames_count):
				frame_info = self.frame_info_list[i]
				if "screen" in frame_info:
					screen = '\n'.join(frame_info["screen"])
					screen_file.write(screen)
					if flags.save_episode_gif:
						filename = episode_directory+'/screens/frame'+str(i)+'.jpg'
						plt.ascii_image(screen, filename)
						screen_filenames.append(filename)
				elif "rgb" in frame_info:
					filename = episode_directory+'/screens/frame'+str(i)+'.jpg'
					plt.rgb_array_image(frame_info["rgb"], filename)
					screen_filenames.append(filename)
		# Heatmap
		if flags.save_episode_heatmap:
			os.mkdir(episode_directory+'/heatmaps')
			heatmap_filenames = []
			for i in range(frames_count):
				frame_info = self.frame_info_list[i]
				if "heatmap" in frame_info:
					filename = episode_directory+'/heatmaps/frame'+str(i)+'.jpg'
					plt.heatmap(heatmap=frame_info["heatmap"], figure_file=filename)
					heatmap_filenames.append(filename)
			# Combine Heatmap and Screen
			os.mkdir(episode_directory+'/heatmap-screens')
			heatmap_screen_filenames = []
			i = 0
			for (heatmap_filename, screen_filename) in zip(heatmap_filenames, screen_filenames):
				filename = episode_directory+'/heatmap-screens/frame'+str(i)+'.jpg'
				plt.combine_images(images_list=[heatmap_filename, screen_filename], file_name=filename)
				heatmap_screen_filenames.append(filename)
				i+=1
		# Gif
		if flags.save_episode_gif:
			filenames = heatmap_screen_filenames if flags.save_episode_heatmap else screen_filenames
			plt.make_gif(file_list=filenames, gif_path=episode_directory + ('/heatmap-screens.gif' if flags.save_episode_heatmap else '/screen.gif'))
		
	def log(self, global_t, step):
		# Speed
		self.print_speed(global_t, step)

		if self.terminal:
			# Statistics
			self.print_statistics()
			# Frames
			if flags.show_all_episodes:
				self.print_frames(global_t)
			elif flags.show_best_episodes:
				if self.episode_reward > self.max_reward:
					self.max_reward = self.episode_reward
					self.print_frames(global_t)
				
	# run simulations
	def run_batch(self):
		if self.train: # Copy weights from shared to local
			self.local_network.sync()
		self.local_network.reset_batch()
			
		step = 0
		while step < flags.max_batch_size and not self.terminal:
			step += 1
			policy, value, action, reward, self.terminal = self.local_network.act( 
				policy_to_action_function=self.environment.choose_action, 
				act_function=self.environment.process, 
				state=self.environment.last_state, 
				concat=self.environment.get_last_action_reward() 
			)
			self.episode_reward += reward
			
			if flags.show_best_episodes or flags.show_all_episodes:
				self.frame_info_list.append( self.environment.get_frame_info(value_estimator_network=self.local_network) )
			
		if self.terminal: # an episode has terminated
			self.terminated_episodes += 1
			
		if self.train: # train using batch
			state = self.environment.last_state if not self.terminal else None # bootstrap
			concat = self.environment.get_last_action_reward() if not self.terminal else None
			self.local_network.compute_cumulative_reward(state=state, concat=concat)
			self.local_network.process_batch()
		return step

	def process(self, global_t=0):
		try:
			if self.terminal:
				self.prepare()
			step = self.run_batch()
			self.log(global_t, step)			
			return step
		except:
			traceback.print_exc()
			self.terminal = True
			return 0