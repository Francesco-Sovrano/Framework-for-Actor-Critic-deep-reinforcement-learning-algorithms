# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import signal
import os
import logging
import time
import json
import pickle
import h5py
import tarfile
import shutil

import options
options.build("training")
flags = options.get()
if not flags.use_gpu:
	# hide gpus so keras can't use them
	# this must be done before importing either keras or tensorflow
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.optimizers import RMSprop
import tensorflow as tf

from environment.environment import Environment
from model.model import MultiAgentModel
from training.trainer import Trainer
from lib.utils import log_uniform
from lib.optimizers import CustomRMSprop


graph = tf.get_default_graph()


class Application(object):
	def __init__(self):
		os.makedirs(flags.log_dir + '/performance', exist_ok=True)
		os.makedirs(flags.log_dir + '/screenshots', exist_ok=True)
		os.makedirs(flags.checkpoint_dir + '/optimizers', exist_ok=True)

		self.reward_logger = logging.getLogger('results')
		hdlr = logging.FileHandler(flags.log_dir + '/results.log')
		formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
		hdlr.setFormatter(formatter)
		self.reward_logger.addHandler(hdlr) 
		self.reward_logger.setLevel(logging.DEBUG)
		self.next_save_steps = flags.save_interval_step

		self.environment = Environment.create_environment(flags.env_type, -1)

		self.global_weigths = []
		self.trainers = []
		""":type : list[Trainer]"""
		self.train_threads = []
		""":type : list[threading.Thread]"""
		# when a thread is started we must wait until it has built its model before starting a new one
		# otherwise the program execution hangs
		# this is the event used to synchronize such behaviour
		self.start_event = threading.Event()
		# after building their networks and before starting their computations, the threads must wait for all the
		# other threads to build their networks, otherwise the program execution is likely to hang
		# this is the condition used to synchronize such behaviour
		self.start_cond = threading.Condition()
		self.start_cond.waiting = 0

		# the threads can't be stopped and restarted when saving (because of keras and tensorflow sessions)
		# so these variables will be used for pausing threads when saving
		self.pause_event = threading.Event()
		self.pause_cond = threading.Condition()
		self.pause_cond.waiting = 0

		self.initial_learning_rate = log_uniform(flags.initial_alpha_low, flags.initial_alpha_high, flags.initial_alpha_log_rate)
		self.start_time = 0
		self.wall_t = 0.0
		self.global_t = 0
		self.terminate_reqested = False

		self.device = "/cpu:0"
		if flags.use_gpu:
			self.device = "/gpu:0"
	
	def train_function(self, parallel_index, preparing):
		""" Train each environment. """

		# create tensorflow session and graph for the thread
		# session and graph can't be saved and used elsewhere, so the threads can't terminate and restart
		with tf.Session(graph=tf.Graph()) as sess:

			print(threading.current_thread().getName(), "starting", flush=True)

			trainer = self.build_trainer(parallel_index)
			self.trainers.append(trainer)

			if preparing:
				trainer.prepare()

			# wake up main thread
			self.start_event.set()

			# wait until other threads built their networks
			# N.B. this is necessary in order to avoid hanging the program execution
			with self.start_cond:
				self.start_cond.waiting += 1
				self.start_cond.notify_all()
			with self.start_cond:
				self.start_cond.wait_for(lambda : self.start_cond.waiting == flags.parallel_size)

			print(threading.current_thread().getName(), "going", flush=True)

			# set start_time
			trainer.set_start_time(self.start_time)

			while True:
				# pauses if it's time to save
				self.thread_check_saving_pause()

				if self.terminate_reqested or self.global_t > flags.max_time_step:
					trainer.stop()
					if parallel_index == 0:
						self.thread_save()
					self.thread_stop(trainer)
					break
				if parallel_index == 0 and self.global_t > self.next_save_steps:
					# Save checkpoint
					self.thread_save()

				diff_global_t = trainer.process(self.global_t)
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

	def run(self):
		try:
			# try to load check point
			self.load_checkpoint(raise_exc=True)
		except FileNotFoundError:
			self.init_global_weights()

		# this will let threads go through .thread_check_saving_pause() without pausing
		self.pause_event.set()

		# set start time
		self.start_time = time.time() - self.wall_t

		print('Press Ctrl+C to stop gracefully')
		signal.signal(signal.SIGINT, self.signal_handler)

		# init and run training threads
		for i in range(flags.parallel_size):
			# run one at a time
			t = threading.Thread(target=self.train_function, args=(i,True), name='thread-%s' % i)
			t.start()
			# wait until the thread built its model
			# N.B. if this waiting is not performed the program execution will most likely hang
			self.start_event.wait()
			self.train_threads.append(t)
			self.start_event.clear()

		signal.pause()

	def signal_handler(self, signal, frame):
		print('You pressed Ctrl+C!')
		self.terminate_reqested = True
		
	def build_network(self, id):
		"""
		Creates an agent model.
		N.B. If the model will be used in a thread it must be created from within it:
		creation and use must be under the same "with tf.Session(graph=tf.Graph()) as sess:".
		"""
		environment = self.environment
		state_shape = environment.get_state_shape()
		agents_count = environment.get_situations_count()
		action_size = environment.get_action_size()
		network = MultiAgentModel(id, state_shape, agents_count, action_size, flags.local_t_max, self.device)
		return network
		
	def build_trainer(self, id):
		"""
		Creates a Trainer with an agent model.
		N.B. If the trainer will be used in a thread it must be created from within it:
		creation and use must be under the same "with tf.Session(graph=tf.Graph()) as sess:".
		"""
		local_network = self.build_network(id)
		thread_name = threading.current_thread().getName()
		opt_fname = '%s/optimizers/%s.hdf5' % (flags.checkpoint_dir, thread_name)
		try:
			with h5py.File(opt_fname, mode='r') as hdf5_file:
				local_network.load_optimizers(hdf5_file, custom_objects={'CustomRMSprop': CustomRMSprop})
		except OSError:
			# no save file found
			local_network.compile(self._make_optimizer)
		trainer = Trainer(id, self.global_weigths, local_network, flags.env_type, flags.local_t_max, flags.gamma, flags.max_time_step, self.device)
		return trainer

	def init_global_weights(self):
		dummy_net = self.build_network(-1)
		self.global_weigths = dummy_net.get_weights()

	def _make_optimizer(self):
		lr = self.initial_learning_rate
		return CustomRMSprop(learning_rate=lr, decay=flags.rmsp_alpha, momentum=0.0, epsilon=flags.rmsp_epsilon,
							 clipnorm=flags.grad_norm_clip, max_iterations=flags.max_time_step)

	def thread_check_saving_pause(self):
		"""
		If a saving pause was requested, pauses the calling thread execution.
		N.B. the trainer should be the one created by the current thread, otherwise keras will raise errors

		:return: whether the thread paused
		"""
		if not self.pause_event.is_set():
			print(threading.current_thread().getName(), "waiting")
			# notify the thread that requested the pause
			with self.pause_cond:
				self.pause_cond.waiting += 1
				self.pause_cond.notify()
			self.pause_event.wait()
			with self.pause_cond:
				self.pause_cond.waiting -= 1
			return True
		return False

	def pause_threads(self):
		"""
		Pauses all threads except the calling one, waiting until they are effectively paused
		"""
		# request pause
		self.pause_event.clear()
		# wait untill all threads are paused
		with self.pause_cond:
			self.pause_cond.wait_for(lambda : self.pause_cond.waiting == flags.parallel_size -1)

	def thread_stop(self, trainer):
		"""
		Saves the thread optimizer and informs the thread supposed to save a checkpoint that this thread is terminating

		:param Trainer trainer:
		"""
		with self.pause_cond:
			self.thread_save_optimizer(trainer)
			self.pause_cond.waiting += 1
			self.pause_cond.notify()
		print(threading.current_thread().getName(), "terminating", flush=True)

	def thread_save_optimizer(self, trainer):
		"""
		Saves the optimizer of the network of the supplied trainer
		N.B. the trainer should be the one created by the current thread, otherwise keras will raise errors

		:type trainer: Trainer
		"""
		thread_name = threading.current_thread().getName()
		opt_fname = '%s/optimizers/%s.hdf5' % (flags.checkpoint_dir, thread_name)
		with h5py.File(opt_fname, mode='w') as hdf5_file:
			trainer.local_network.save_optimizers(hdf5_file)
			
	def thread_save(self):
		""" Save checkpoint. Called from thread-0.

		:type trainer: Trainer
		"""
		print("pausing to save", flush=True)
		self.pause_threads()

		print('Start saving.')

		self.save_checkpoint()

		print('End saving.')

		# resume threads
		self.pause_event.set()
		self.next_save_steps += flags.save_interval_step

	def save_checkpoint(self):
		# Create dir
		os.makedirs(flags.checkpoint_dir, exist_ok=True)

		# Write wall time and global_t
		wall_t = time.time() - self.start_time
		data = {'wall_t': wall_t, 'global_t': self.global_t}
		data_fname = '%s/data' % flags.checkpoint_dir
		# will save 2 copies, one in 'data' and the other in 'data-global_t'
		# the copy in 'data' is the one that will be loaded, the other is kept for history
		for fname in [data_fname, data_fname + '-%s' % self.global_t]:
			with open(fname, 'w') as df:
				json.dump(data, df)

		# save global weigths
		weights_fname = '%s/weights.pkl' % flags.checkpoint_dir
		with open(weights_fname, mode='wb') as f:
			pickle.dump(self.global_weigths, f)

		# optimizers are saved only when terminating in .thread_stop()

	def load_checkpoint(self, raise_exc=False):
		print("Attempting to load checkpoint...", end=' ')
		try:
			# load wall time and global_t
			data_fname = '%s/data' % flags.checkpoint_dir
			with open(data_fname) as df:
				data = json.load(df)
			self.global_t = data['global_t']
			self.wall_t = data['wall_t']
			print(">>> global step set: ", self.global_t)
			self.next_save_steps = (self.global_t + flags.save_interval_step) // flags.save_interval_step * flags.save_interval_step

			# load global weights
			weights_fname = '%s/weights.pkl' % flags.checkpoint_dir
			with open(weights_fname, mode='rb') as f:
				self.global_weigths = pickle.load(f)

			# optimizers will be loaded in .build_network()

		except FileNotFoundError as exc:
			print("none found")
			if raise_exc:
				raise exc


if __name__ == '__main__':
	app = Application()
	app.run()
