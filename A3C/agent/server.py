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
from multiprocessing import Queue

from environment.environment import Environment
from agent.client import Worker
import utils.plots as plt
from agent.manager import *

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
		self.session = tf.Session(config=config)
		self.global_step = 0
		self.stop_requested = False
		self.terminate_reqested = False
		self.build_network()
			
	def build_network(self):
		# global network
		self.global_network = Worker(thread_index=0, session=self.session, global_network=None, device=self.device).local_network
		# local networks
		self.trainers = []
		for i in range(flags.parallel_size):
			self.trainers.append( Worker(thread_index=i+1, session=self.session, global_network=self.global_network, device=self.device) )
		# initialize variables
		self.session.run(tf.global_variables_initializer()) # do it before loading checkpoint
		# load checkpoint
		self.load_checkpoint()
		# print graph summary
		tf.summary.FileWriter('summary', self.session.graph).close()
		
	def test_function(self, tester, count):
		results = []
		tester.set_start_time(self.start_time)
		for _ in range(count):
			tester.prepare()
			while not tester.terminal:
				tester.process()
			results.append( tester.environment.get_test_result() )
		return results

	def test(self):
		result_file = '{}/test_results_{}.log'.format(flags.log_dir,self.global_step)
		if os.path.exists(result_file):
			print('Test results already produced and evaluated for {}'.format(result_file))
			return
			
		print('Start testing')
		testers = []
		threads = []
		result_queue = Queue()
		for i in range(flags.parallel_size): # parallel testing
			tester = Worker(thread_index=-i-1, session=self.session, global_network=self.global_network, device=self.device, training=False)
			thread = threading.Thread(target=lambda q, args: q.put(test_function(*args)), args=(result_queue,(tester,tester.environment.get_test_size())))
			thread.start()
			threads.append(thread)
			testers.append(tester)
		time.sleep(5)
		for thread in threads: # wait for all threads to end
			thread.join()
		print('End testing')
		# get overall statistics
		info = self.get_global_statistics(clients=testers)
		# write results to file
		stats_file = '{}/test_statistics.log'.format(flags.log_dir)
		with open(stats_file, "a", encoding="utf-8") as file: # write stats to file
			file.write(str(["{}={}".format(key,value) for key,value in sorted(info.items(), key=lambda t: t[0])]))
		print('Test statistics saved in {}'.format(stats_file))
		with open(result_file, "w", encoding="utf-8") as file: # write results to file
			while not result_queue.empty():
				result = result_queue.get()
				for line in result:
					file.write(line)
		print('Test results saved in {}'.format(result_file))
		return testers[0].environment.evaluate_test_results(result_file)

	def train_function(self, parallel_index):
		""" Train each environment. """
		trainer = self.trainers[parallel_index]
		# set start_time
		trainer.set_start_time(self.start_time)
	
		while True:
			if flags.synchronize_threads:
				while self.sync_event.is_set(): # wait for other threads to start
					time.sleep(flags.synchronization_sleep)
					# print(parallel_index, "waiting")
	
			diff_global_step = trainer.process(self.global_step)
			self.global_step += diff_global_step
			# print global statistics
			if trainer.terminal:
				info = self.get_global_statistics(clients=self.trainers)
				if info:
					info_str = "<{}> {}".format(self.global_step, ["{}={}".format(key,value) for key,value in sorted(info.items(), key=lambda t: t[0])])
					self.training_logger.info(info_str) # Print statistics
				if parallel_index == 0:
					sys.stdout.flush() # force print immediately what is in output buffer
			if flags.synchronize_threads:
				# print(parallel_index, 'end')
				with self.sync_lock:
					self.sync_count += 1 # thread p ended
					# print('sum before', self.sync_count)
					if self.sync_count == flags.parallel_size: # all threads are ended
						self.sync_event.set() # start synching
				event_is_set = self.sync_event.wait()
				# print('event set: ', event_is_set, parallel_index)
				with self.sync_lock:
					self.sync_count -= 1 # thread p can start
					# print('sum after', self.sync_count)
					if self.sync_count == 0: # all threads can start
						self.sync_event.clear() # synching completed
			# do it after synching threads
			if self.stop_requested:
				return
			if self.terminate_reqested:
				trainer.stop()
				if parallel_index == 0:
					self.save()
				return
			if self.global_step > flags.max_time_step:
				trainer.stop()
				return
			if self.global_step > self.next_save_steps:
				if parallel_index == 0: # Save checkpoint
					self.save()
				else:
					return		

	def get_global_statistics(self, clients):
		dictionaries = [client.stats for client in clients if client.terminated_episodes >= flags.match_count_for_evaluation]
		used_clients = len(dictionaries) # ignore the first flags.match_count_for_evaluation objects from data, because they are too noisy
		if used_clients < 1:
			return {}
		return {k: sum(d[k] for d in dictionaries if k in d)/used_clients for k in dictionaries[0]}
		
	def train(self):
		# run training threads
		self.train_threads = [threading.Thread(target=self.train_function, args=(i,)) for i in range(flags.parallel_size)]
		signal.signal(signal.SIGINT, self.signal_handler)
		# set start time
		self.start_time = time.time() - self.elapsed_time
		if flags.synchronize_threads: # build synchronization vector
			self.sync_event = threading.Event()
			self.sync_lock = threading.Lock()
			self.sync_count = 0
		for t in self.train_threads:
			t.start()
		print('Press Ctrl+C to stop')
		signal.pause()
	
	def load_checkpoint(self):
		# init or load checkpoint with saver
		self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
		checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.session, checkpoint.model_checkpoint_path)
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
		self.session.graph.finalize()
			
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
		self.saver.save(self.session, flags.checkpoint_dir + '/checkpoint', global_step=self.global_step)
		self.save_important_information(flags.checkpoint_dir + '/{}.pkl'.format(self.global_step))
		print('Checkpoint saved in ' + flags.checkpoint_dir)
		
		# Test
		if flags.test_after_saving:
			self.test()
		
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
		# Experience replay
		if flags.replay_ratio > 0:
			persistent_memory["experience_buffers"] = [None for _ in range(trainers_count)]
		# Reward prediction
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