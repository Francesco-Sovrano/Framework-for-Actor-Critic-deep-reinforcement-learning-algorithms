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

from environment.environment import Environment
from model.multi_agent_model import MultiAgentModel
from work.trainer import Trainer
from work.rmsprop_applier import RMSPropApplier

import options
options.build("training")
flags = options.get()

def log_uniform(lo, hi, rate):
	log_lo = math.log(lo)
	log_hi = math.log(hi)
	v = log_lo * (1-rate) + log_hi * rate
	return math.exp(v)


class Application(object):
	def __init__(self):
		self.reward_logger = logging.getLogger('results')
		hdlr = logging.FileHandler(flags.log_dir + '/results.log')
		formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
		hdlr.setFormatter(formatter)
		self.reward_logger.addHandler(hdlr) 
		self.reward_logger.setLevel(logging.DEBUG)
	
	def train_function(self, parallel_index, preparing):
		""" Train each environment. """
		
		trainer = self.trainers[parallel_index]
		if preparing:
			trainer.prepare()
		
		# set start_time
		trainer.set_start_time(self.start_time)
	
		while True:
			if self.stop_requested:
				break
			if self.terminate_reqested:
				trainer.stop()
				self.save()
				break
			if self.global_t > flags.max_time_step:
				trainer.stop()
				break
			if parallel_index == 0 and self.global_t > self.next_save_steps:
				# Save checkpoint
				self.save()
	
			diff_global_t = trainer.process(self.sess, self.global_t, self.summary_writer, self.summary_op, self.score_input)
			self.global_t += diff_global_t
			
			# print global statistics
			if trainer.episode_steps == 0:
				info = {}
				for t in self.trainers:
					for key in t.stats:
						if not info.get(key):
							info[key] = 0
						info[key] += t.stats[key]
				log_str = ""
				for key in info:
					log_str += " " + key + "=" + str(info[key]/len(self.trainers))
				self.reward_logger.info( log_str )
				sys.stdout.flush()

	def run(self):
		self.device = "/cpu:0"
		if flags.use_gpu:
			self.device = "/gpu:0"
			
		# prepare session
		config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
		if flags.use_gpu:
			config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)
		
		self.global_t = 0
		self.stop_requested = False
		self.terminate_reqested = False
		
		self.build_network()
		
		# summary for tensorboard
		self.score_input = tf.placeholder(tf.int32)
		tf.summary.scalar("score", self.score_input)
		self.summary_op = tf.summary.merge_all()
		self.summary_writer = tf.summary.FileWriter(flags.event_dir, self.sess.graph)
	
		# run training threads
		self.train_threads = []
		for i in range(flags.parallel_size):
			self.train_threads.append(threading.Thread(target=self.train_function, args=(i,True)))
			
		signal.signal(signal.SIGINT, self.signal_handler)
	
		# set start time
		self.start_time = time.time() - self.wall_t
	
		for t in self.train_threads:
			t.start()
	
		print('Press Ctrl+C to stop')
		signal.pause()
		
	def build_network(self):
		learning_rate_input = tf.placeholder("float")
		grad_applier = self.build_global_network(learning_rate_input)
		self.build_local_networks(learning_rate_input, grad_applier)
		self.sess.run(tf.global_variables_initializer()) # do it before loading checkpoint
		self.load_checkpoint()
		
	def build_global_network(self, learning_rate_input):
		environment = Environment.create_environment(flags.env_type, -1)
		state_shape = environment.get_state_shape()
		agents_count = environment.get_situations_count()
		action_size = environment.get_action_size()
		self.global_network = MultiAgentModel( -1, state_shape, agents_count, action_size, flags.entropy_beta, self.device )
		return RMSPropApplier(learning_rate = learning_rate_input, decay = flags.rmsp_alpha, momentum = 0.0, epsilon = flags.rmsp_epsilon, clip_norm = flags.grad_norm_clip, device = self.device)
		
	def build_local_networks(self, learning_rate_input, grad_applier):
		initial_learning_rate = log_uniform(flags.initial_alpha_low, flags.initial_alpha_high, flags.initial_alpha_log_rate)
		self.trainers = []
		for i in range(flags.parallel_size):
			trainer = Trainer(i, self.global_network, initial_learning_rate, learning_rate_input, grad_applier, flags.env_type, flags.entropy_beta,flags.local_t_max,flags.gamma, flags.max_time_step, self.device)
			self.trainers.append(trainer)
	
	def load_checkpoint(self):
		# init or load checkpoint with saver
		self.saver = tf.train.Saver(self.global_network.get_vars())
		checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print("checkpoint loaded:", checkpoint.model_checkpoint_path)
			tokens = checkpoint.model_checkpoint_path.split("-")
			# set global step
			self.global_t = int(tokens[1])
			print(">>> global step set: ", self.global_t)
			# set wall time
			wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(self.global_t)
			with open(wall_t_fname, 'r') as f:
				self.wall_t = float(f.read())
				self.next_save_steps = (self.global_t + flags.save_interval_step) // flags.save_interval_step * flags.save_interval_step
		else:
			print("Could not find old checkpoint")
			# set wall time
			self.wall_t = 0.0
			self.next_save_steps = flags.save_interval_step
			
	def save(self):
		""" Save checkpoint. 
		Called from therad-0.
		"""
		self.stop_requested = True
	
		# Wait for all other threads to stop
		for (i, t) in enumerate(self.train_threads):
			if i != 0:
				t.join()
	
		# Save
		if not os.path.exists(flags.checkpoint_dir):
			os.mkdir(flags.checkpoint_dir)
	
		# Write wall time
		wall_t = time.time() - self.start_time
		wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(self.global_t)
		with open(wall_t_fname, 'w') as f:
			f.write(str(wall_t))
	
		print('Start saving.')
		self.saver.save(self.sess, flags.checkpoint_dir + '/' + 'checkpoint', global_step = self.global_t)
		print('End saving.')	
	
		if not self.terminate_reqested:
			self.stop_requested = False
			self.next_save_steps += flags.save_interval_step
			# Restart other threads
			for i in range(flags.parallel_size):
				if i != 0:
					thread = threading.Thread(target=self.train_function, args=(i,False))
					self.train_threads[i] = thread
					thread.start()
		
	def signal_handler(self, signal, frame):
		print('You pressed Ctrl+C!')
		self.terminate_reqested = True

def main(argv):
	app = Application()
	app.run()

if __name__ == '__main__':
	tf.app.run()