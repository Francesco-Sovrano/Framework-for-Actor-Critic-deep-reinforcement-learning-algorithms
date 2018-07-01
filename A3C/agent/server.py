# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import threading

import signal
import math
import os
import logging
import time
import sys
import pickle

from environment.environment import Environment
from agent.client import Worker
import agent.plots as plt

import options
flags = options.get()

import numpy as np

class Application(object):
	def __init__(self):
		self.train_logfile = flags.log_dir + '/train_results.log'
		# Training logger
		self.training_logger = logging.getLogger('results')
		if not os.path.isdir(flags.log_dir):
			os.mkdir(flags.log_dir)
		hdlr = logging.FileHandler(self.train_logfile)
		formatter = logging.Formatter('%(asctime)s %(message)s')
		hdlr.setFormatter(formatter)
		self.training_logger.addHandler(hdlr) 
		self.training_logger.setLevel(logging.DEBUG)
		# Initialize network
		self.device = "/cpu:0"
		if flags.use_gpu:
			self.device = "/gpu:0"
		config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True) # prepare session
		if flags.use_gpu:
			config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)
		self.global_t = 0
		self.stop_requested = False
		self.terminate_reqested = False
		self.build_network()
			
	def build_network(self):
		# global network
		self.global_network = Worker(thread_index=0, session=self.sess, global_network=None, device=self.device).local_network
		# local networks
		self.trainers = []
		for i in range(flags.parallel_size):
			self.trainers.append( Worker(thread_index=i+1, session=self.sess, global_network=self.global_network, device=self.device) )
		# initialize variables
		self.sess.run(tf.global_variables_initializer()) # do it before loading checkpoint
		# load checkpoint
		self.load_checkpoint()
		
	def test_function(self, tester, count):
		for _ in range(count):
			tester.prepare()
			while not tester.terminal:
				tester.process()

	def test(self):
		print('Start testing')
		testers = []
		threads = []
		for i in range(flags.parallel_size): # parallel testing
			tester = Worker(thread_index=-i, session=self.sess, global_network=self.global_network, device=self.device, train=False)
			thread = threading.Thread(target=self.test_function, args=(tester,flags.match_count_for_evaluation//flags.parallel_size))
			thread.start()
			threads.append(thread)
			testers.append(tester)
		time.sleep(5)
		for thread in threads: # wait for all threads to end
			thread.join()
		# get overall statistics
		info = self.get_global_statistics(clients=testers)
		# write results to file
		with open(flags.log_dir + '/test_results.log', "w", encoding="utf-8") as file:
			file.write(str([key + "=" + str(value) for key, value in sorted(info.items(), key=lambda t: t[0])]))
		print('End testing')
		print('Test result saved in ' + flags.log_dir + '/test_results.log')

	def train_function(self, parallel_index):
		""" Train each environment. """
		trainer = self.trainers[parallel_index]
		# set start_time
		trainer.set_start_time(self.start_time)
	
		while True:
			if self.stop_requested:
				break
			if self.terminate_reqested:
				trainer.stop()
				if parallel_index == 0:
					self.save()
				break
			if self.global_t > flags.max_time_step:
				trainer.stop()
				break
			if parallel_index == 0 and self.global_t > self.next_save_steps:
				# Save checkpoint
				self.save()
	
			diff_global_t = trainer.process(self.global_t)
			self.global_t += diff_global_t
			
			# print global statistics
			if trainer.terminal:
				info = self.get_global_statistics(clients=self.trainers)
				if info:
					self.training_logger.info( str([key + "=" + str(value) for key, value in sorted(info.items(), key=lambda t: t[0])]) ) # Print statistics
				sys.stdout.flush() # force print immediately what is in output buffer
				
	def get_global_statistics(self, clients):
		info = {}
		unused_clients = 0
		for client in clients:
			if client.terminated_episodes < flags.match_count_for_evaluation: # ignore the first flags.match_count_for_evaluation objects from data, because they are too noisy
				unused_clients += 1
				continue
			for key in client.stats:
				if not info.get(key):
					info[key] = 0
				info[key] += client.stats[key]
		if unused_clients < len(clients):
			for key in info:
				info[key] /= len(clients) - unused_clients
		return info
		
	def train(self):
		# run training threads
		self.train_threads = []
		for i in range(flags.parallel_size):
			self.train_threads.append(threading.Thread(target=self.train_function, args=(i,)))
		signal.signal(signal.SIGINT, self.signal_handler)
		# set start time
		self.start_time = time.time() - self.wall_t
		for t in self.train_threads:
			t.start()
		print('Press Ctrl+C to stop')
		signal.pause()
	
	def load_checkpoint(self):
		# init or load checkpoint with saver
		self.saver = tf.train.Saver(self.global_network.get_vars())
		checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			tokens = checkpoint.model_checkpoint_path.split("-")
			# set global step
			self.global_t = int(tokens[1])
			print(">>> global step set: ", self.global_t)
			# set wall time
			wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(self.global_t)
			with open(wall_t_fname, 'r') as f:
				self.wall_t = float(f.read())
				self.next_save_steps = (self.global_t + flags.save_interval_step) // flags.save_interval_step * flags.save_interval_step
			self.load_important_information(flags.checkpoint_dir + '/{0}.pkl'.format(self.global_t))
			print("Checkpoint loaded: ", checkpoint.model_checkpoint_path)
		else:
			# set wall time
			self.wall_t = 0.0
			self.next_save_steps = flags.save_interval_step
			print("Could not find old checkpoint")
			
	def save(self):
		""" Save checkpoint. 
		Called from thread-0.
		"""
		self.stop_requested = True
		for (i, t) in enumerate(self.train_threads): # Wait for all other threads to stop
			if i != 0: # cannot join current thread
				t.join()
	
		# Save
		if not os.path.exists(flags.checkpoint_dir):
			os.mkdir(flags.checkpoint_dir)
	
		# Write wall time
		wall_t = time.time() - self.start_time
		wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(self.global_t)
		with open(wall_t_fname, 'w') as f:
			f.write(str(wall_t))
	
		# Print plot
		if flags.compute_plot_when_saving:
			plt.plot_files(log_files=[self.train_logfile], figure_file=flags.log_dir + '/train_plot.jpg')
		
		# Save Checkpoint
		print('Start saving..')
		self.saver.save(self.sess, flags.checkpoint_dir + '/checkpoint', global_step=self.global_t)
		self.save_important_information(flags.checkpoint_dir + '/{0}.pkl'.format(self.global_t))
		print('Checkpoint saved in ' + flags.checkpoint_dir)
	
		if not self.terminate_reqested:
			self.stop_requested = False
			self.next_save_steps += flags.save_interval_step
			# Restart other threads
			for i in range(flags.parallel_size):
				if i != 0: # current thread is already running
					thread = threading.Thread(target=self.train_function, args=(i,))
					self.train_threads[i] = thread
					thread.start()
					
	def save_important_information(self, path):
		persistent_memory = {}
		persistent_memory["train_count_matrix"] = []
		if flags.replay_ratio > 0:
			persistent_memory["experience_buffers"] = []
		for trainer in self.trainers:
			# train counters
			tc = []
			for model in trainer.local_network.model_list:
				tc.append(model.train_count)
			persistent_memory["train_count_matrix"].append(tc)
			# experience buffer
			if flags.replay_ratio > 0:
				persistent_memory["experience_buffers"].append(trainer.local_network.experience_buffer)
		with open(path, 'wb') as f:
			pickle.dump(persistent_memory, f, pickle.HIGHEST_PROTOCOL)
			
	def load_important_information(self, path):
		with open(path, 'rb') as f:
			persistent_memory = pickle.load(f)
			i = 0
			for trainer in self.trainers:
				# train counters
				j=0
				for model in trainer.local_network.model_list:
					model.train_count = persistent_memory["train_count_matrix"][i][j]
					j+=1
				# experience buffer
				if flags.replay_ratio > 0:
					trainer.local_network.experience_buffer = persistent_memory["experience_buffers"][i]
				i+=1
		
	def signal_handler(self, signal, frame):
		print('You pressed Ctrl+C!')
		self.terminate_reqested = True