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
import _pickle as pickle # CPickle

from environment.environment import Environment
from agent.client import Worker
import utils.plots as plt
# import gc

import options
flags = options.get()
import numpy as np

class Application(object):
	# __slots__ = ('training_logger','train_logfile','sess','global_step','stop_requested','terminate_reqested','lock','device',
	#				 'global_network','trainers','saver','elapsed_time','next_save_steps','train_threads')
					
	def __init__(self):
		if not os.path.isdir(flags.log_dir):
			os.mkdir(flags.log_dir)
		self.train_logfile = flags.log_dir + '/train_results.log'
		# Training logger
		self.training_logger = logging.getLogger('results')
		hdlr = logging.FileHandler(self.train_logfile)
		hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
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
		self.global_step = 0
		self.stop_requested = False
		self.terminate_reqested = False
		self.build_network()
		self.lock = threading.Lock()
			
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
		# print graph summary
		tf.summary.FileWriter('summary', self.sess.graph).close()
		
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
			tester = Worker(thread_index=-i, session=self.sess, global_network=self.global_network, device=self.device, training=False)
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
			file.write(str(["{}={}".format(key,value) for key,value in sorted(info.items(), key=lambda t: t[0])]))
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
			if self.global_step > flags.max_time_step:
				trainer.stop()
				break
			if parallel_index == 0 and self.global_step > self.next_save_steps:
				# Save checkpoint
				self.save()
	
			diff_global_step = trainer.process(self.global_step)
			with self.lock:
				self.global_step += diff_global_step			
				# print global statistics
				if trainer.terminal:
					info = self.get_global_statistics(clients=self.trainers)
					if info:
						info_str = "<{}> {}".format(self.global_step, ["{}={}".format(key,value) for key,value in sorted(info.items(), key=lambda t: t[0])])
						self.training_logger.info(info_str) # Print statistics
					sys.stdout.flush() # force print immediately what is in output buffer

	def get_global_statistics(self, clients):
		dictionaries = [client.stats for client in clients if client.terminated_episodes >= flags.match_count_for_evaluation]
		used_clients = len(dictionaries) # ignore the first flags.match_count_for_evaluation objects from data, because they are too noisy
		if used_clients < 1:
			return {}
		return {k: sum(d[k] for d in dictionaries)/used_clients for k in dictionaries[0]}
		
	def train(self):
		# run training threads
		self.train_threads = []
		for i in range(flags.parallel_size):
			self.train_threads.append(threading.Thread(target=self.train_function, args=(i,)))
		signal.signal(signal.SIGINT, self.signal_handler)
		# set start time
		self.start_time = time.time() - self.elapsed_time
		for t in self.train_threads:
			t.start()
		print('Press Ctrl+C to stop')
		signal.pause()
	
	def load_checkpoint(self):
		# init or load checkpoint with saver
		self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
		checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			tokens = checkpoint.model_checkpoint_path.split("-")
			# set global step
			self.global_step = int(tokens[1])
			print(">>> global step set: ", self.global_step)
			# set wall time
			elapsed_time_fname = flags.checkpoint_dir + '/' + 'elapsed_time.' + str(self.global_step)
			with open(elapsed_time_fname, 'r') as f:
				self.elapsed_time = float(f.read())
				self.next_save_steps = (self.global_step + flags.save_interval_step) // flags.save_interval_step * flags.save_interval_step
			self.load_important_information(flags.checkpoint_dir + '/{0}.pkl'.format(self.global_step))
			print("Checkpoint loaded: ", checkpoint.model_checkpoint_path)
		else:
			# set wall time
			self.elapsed_time = 0.0
			self.next_save_steps = flags.save_interval_step
			print("Could not find old checkpoint")
		# self.sess.graph.finalize()
			
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
		elapsed_time = time.time() - self.start_time
		elapsed_time_fname = flags.checkpoint_dir + '/' + 'elapsed_time.' + str(self.global_step)
		with open(elapsed_time_fname, 'w') as f:
			f.write(str(elapsed_time))
	
		# Print plot
		if flags.compute_plot_when_saving:
			plt.plot_files(log_files=[self.train_logfile], figure_file=flags.log_dir + '/train_plot.jpg')
		
		# Save Checkpoint
		print('Start saving..')
		self.saver.save(self.sess, flags.checkpoint_dir + '/checkpoint', global_step=self.global_step)
		self.save_important_information(flags.checkpoint_dir + '/{}.pkl'.format(self.global_step))
		print('Checkpoint saved in ' + flags.checkpoint_dir)
		# gc.collect()
		
		# Restart workers
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
		trainers_count = len(self.trainers)
		persistent_memory = {}
		persistent_memory["train_count_matrix"] = [[] for _ in range(trainers_count)]
		if flags.replay_ratio > 0:
			persistent_memory["experience_buffers"] = [None for _ in range(trainers_count)]
		if flags.predict_reward:
			persistent_memory["reward_prediction_buffers"] = [None for _ in range(trainers_count)]
		for i in range(trainers_count):
			trainer = self.trainers[i]
			# train counters
			train_count_matrix = persistent_memory["train_count_matrix"][i]
			for model in trainer.local_network.model_list:
				train_count_matrix.append(model.train_count)
			# experience buffer
			if flags.replay_ratio > 0:
				persistent_memory["experience_buffers"][i] = trainer.local_network.experience_buffer
			if flags.predict_reward:
				persistent_memory["reward_prediction_buffers"][i] = trainer.local_network.reward_prediction_buffer
		with open(path, 'wb') as file:
			pickle.dump(persistent_memory, file)
			
	def load_important_information(self, path):
		with open(path, 'rb') as file:
			persistent_memory = pickle.load(file)
			
		for (i, trainer) in enumerate(self.trainers):
			# train counters
			for (j, model) in enumerate(trainer.local_network.model_list):
				model.train_count = persistent_memory["train_count_matrix"][i][j]
			# experience buffer
			if flags.replay_ratio > 0:
				trainer.local_network.experience_buffer = persistent_memory["experience_buffers"][i]
			if flags.predict_reward:
				trainer.local_network.reward_prediction_buffer = persistent_memory["reward_prediction_buffers"][i]
		
	def signal_handler(self, signal, frame):
		print('You pressed Ctrl+C!')
		self.terminate_reqested = True