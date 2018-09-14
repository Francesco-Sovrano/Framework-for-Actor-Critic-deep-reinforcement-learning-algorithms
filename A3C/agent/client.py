# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import gc
import traceback
import collections
import os
import logging
import numpy as np
import time
import utils.plots as plt

from environment.environment import Environment
from agent.manager import *
from utils.schedules import LinearSchedule

# get command line args
import options
flags = options.get()

PERFORMANCE_LOG_INTERVAL = 1000

class Worker(object):
	
	def get_model_manager(self):
		if flags.partition_count < 2:
			return "BasicManager"
		return flags.partitioner_type + "Partitioner"
			
	def __init__(self, thread_index, session, global_network, device, training=True):
		self.training = training
		self.thread_index = thread_index
		self.global_network = global_network
		self.device = device
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
		self.environment = Environment.create_environment(flags.env_type, self.thread_index, self.training)
		state_shape = self.environment.get_state_shape()
		action_shape = self.environment.get_action_shape()
		concat_size = self.environment.get_concatenation_size() if flags.use_concatenation else 0
		self.local_network = eval(self.get_model_manager())(
			session=session, 
			device=self.device, 
			id=self.thread_index, 
			action_shape=action_shape, 
			concat_size=concat_size,
			state_shape=state_shape, 
			global_network=self.global_network,
			training=self.training
		)
		self.terminal = True
		self.local_t = 0
		self.prev_local_t = 0
		self.terminated_episodes = 0
		self.stats = {}
		self.batch_schedule = LinearSchedule(flags.max_time_step-flags.steps_before_increasing_batch_size, initial_p=flags.min_batch_size, final_p=flags.min_batch_size)

	def update_statistics(self):
		self.stats = self.environment.get_statistics()
		self.stats.update(self.local_network.get_statistics())

	def prepare(self): # initialize a new episode
		self.terminal = False
		self.episode_reward = 0
		self.environment.reset()
		self.local_network.reset()
		# frame info
		self.frame_info_list = []
		if flags.show_episodes == 'none':
			self.save_frame_info = False
		else:
			self.save_frame_info = flags.show_episodes != 'random' or np.random.random() <= flags.show_episode_probability

	def stop(self): # stop current episode
		self.environment.stop()
		
	def set_start_time(self, start_time):
		self.start_time = start_time
		
	def print_speed(self, global_step, step):
		self.local_t += step
		if (self.thread_index == 1) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
			self.prev_local_t += PERFORMANCE_LOG_INTERVAL
			elapsed_time = time.time() - self.start_time
			steps_per_sec = global_step / elapsed_time
			print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format( global_step, elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))
		
	def print_statistics(self):
		# Update statistics
		self.update_statistics()
		# Print statistics
		self.reward_logger.info( str(["{0}={1}".format(key,value) for key,value in self.stats.items()]) )
		
	def print_frames(self, global_step):
		frames_count = len(self.frame_info_list)
		if frames_count == 0:
			return
		episode_directory = "{}/episodes/reward({})_step({})_thread({})".format(flags.log_dir, self.episode_reward, global_step, self.thread_index)
		os.mkdir(episode_directory)
		
		first_frame = self.frame_info_list[0]
		has_log = "log" in first_frame
		has_screen = "screen" in first_frame
		has_heatmap = "heatmap" in first_frame
		
		gif_filenames = []
		# Log
		if has_log:
			with open(episode_directory + '/episode.log',"w") as screen_file:
				for i in range(frames_count):
					frame_info = self.frame_info_list[i]
					screen_file.write(frame_info["log"])
		# Screen
		if has_screen:
			os.mkdir(episode_directory+'/screens')
			screen_filenames = []
			for i in range(frames_count):
				filename = episode_directory+'/screens/frame'+str(i)+'.jpg'
				
				frame_info = self.frame_info_list[i]
				screen_type = frame_info["screen"]["type"]
				screen_value = frame_info["screen"]["value"]
				if screen_type == 'ASCII':
					plt.ascii_image(screen_value, filename)
				elif screen_type == 'RGB':
					plt.rgb_array_image(screen_value, filename)
				screen_filenames.append(filename)
			gif_filenames = screen_filenames
		# Heatmap
		if has_heatmap:
			os.mkdir(episode_directory+'/heatmaps')
			heatmap_filenames = []
			for i in range(frames_count):
				frame_info = self.frame_info_list[i]
				filename = episode_directory+'/heatmaps/frame'+str(i)+'.jpg'
				plt.heatmap(heatmap=frame_info["heatmap"], figure_file=filename)
				heatmap_filenames.append(filename)
			gif_filenames = heatmap_filenames
		# Combine Heatmap and Screen
		if has_heatmap and has_screen:
			os.mkdir(episode_directory+'/heatmap-screens')
			heatmap_screen_filenames = []
			i = 0
			for (heatmap_filename, screen_filename) in zip(heatmap_filenames, screen_filenames):
				filename = episode_directory+'/heatmap-screens/frame'+str(i)+'.jpg'
				plt.combine_images(images_list=[heatmap_filename, screen_filename], file_name=filename)
				heatmap_screen_filenames.append(filename)
				i+=1
			gif_filenames = heatmap_screen_filenames
		# Gif
		if flags.save_episode_gif and len(gif_filenames) > 0:
			plt.make_gif(file_list=gif_filenames, gif_path=episode_directory+'/episode.gif')
		# gc.collect()
		
	def log(self, global_step, step):
		# Speed
		self.print_speed(global_step, step)

		if self.terminal:
			# Statistics
			self.print_statistics()
			# Frames
			if flags.show_episodes == 'best':
				if self.episode_reward > self.max_reward:
					self.max_reward = self.episode_reward
					self.print_frames(global_step)
			elif self.save_frame_info:
				self.print_frames(global_step)
				
	def get_batch_size(self, global_step):
		if flags.max_batch_size <= flags.min_batch_size or global_step <= flags.steps_before_increasing_batch_size:
			return flags.min_batch_size
		return self.batch_schedule.value(global_step-flags.steps_before_increasing_batch_size)
				
	# run simulations
	def run_batch(self, global_step):
		if self.training: # Copy weights from shared to local
			self.local_network.sync()
			self.local_network.initialize_new_batch()
		step = 0
		max_batch_size = self.get_batch_size(global_step)
		while step < max_batch_size and not self.terminal:
			step += 1
			state=self.environment.last_state
			new_state, value, action, reward, self.terminal, policy = self.local_network.act( 
				act_function=self.environment.process, 
				state=state,
				concat=self.environment.get_concatenation() if flags.use_concatenation else None
			)
			self.episode_reward += sum(reward)
			
			if self.save_frame_info:
				self.frame_info_list.append( self.environment.get_frame_info(network=self.local_network, value=value, action=action, reward=reward, policy=policy) )
			
		if self.terminal: # an episode has terminated
			self.terminated_episodes += 1
			
		if self.training: # train using batch
			if not self.terminal:
				self.local_network.bootstrap(state=new_state, concat=self.environment.get_concatenation() if flags.use_concatenation else None)
			self.local_network.process_batch(global_step)
		return step

	def process(self, global_step=0):
		try:
			if self.terminal:
				self.prepare()
			step = self.run_batch(global_step)
			self.log(global_step, step)
			return step
		except:
			traceback.print_exc()
			self.terminal = True
			return 0