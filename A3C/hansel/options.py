# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def build(option_type):
	"""
	option_type: string
		'training' or 'display' or 'visualize'
	"""
	# Common
	tf.app.flags.DEFINE_boolean("use_gpu", False, "whether to use the GPU")
	tf.app.flags.DEFINE_string("state_generator", "CroppedView_2b_2L_StateGenerator", "the state generator must be a classname from rogueinabox/states.py")
	tf.app.flags.DEFINE_string("reward_generator", "ImprovedStairSeeker_RewardGenerator", "the reward generator must be a classname from rogueinabox/rewards.py")
	
	tf.app.flags.DEFINE_string("env_type", "rogue", "environment type")
	tf.app.flags.DEFINE_string("env_path", "/home/students/francesco.sovrano/Documents/ML/Rogue/rogue5.4.4-ant-r1.1.4/rogue", "the path where to find the game")
	tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint", "checkpoint directory")
	tf.app.flags.DEFINE_string("event_dir", "./events", "events directory")
	tf.app.flags.DEFINE_string("log_dir", "./log", "events directory")
	tf.app.flags.DEFINE_boolean("show_best_screenshots", True, "whether to save the best matches")
	tf.app.flags.DEFINE_boolean("show_all_screenshots", False, "whether to save all the matches")

	tf.app.flags.DEFINE_string("rogueinabox_path", "/home/students/francesco.sovrano/Documents/ML/Rogue", "where to find the package") # to remove!

	# For training
	if option_type == 'training':
		tf.app.flags.DEFINE_integer("parallel_size", 16, "parallel thread size")
		tf.app.flags.DEFINE_integer("steps_per_episode", 500, "number of maximum actions execution per episode")
		tf.app.flags.DEFINE_integer("match_count_for_evaluation", 200, "number of matches used for evaluation scores")
		tf.app.flags.DEFINE_integer("local_t_max", 60, "repeat step size")
		tf.app.flags.DEFINE_float("rmsp_alpha", 0.99, "decay parameter for rmsprop")
		tf.app.flags.DEFINE_float("rmsp_epsilon", 0.1, "epsilon parameter for rmsprop")

		tf.app.flags.DEFINE_string("log_file", "/home/orto/Desktop/MachineLearning/UNREAL/events", "log file directory")
		tf.app.flags.DEFINE_float("initial_alpha_low", 1e-4, "log_uniform low limit for learning rate")
		tf.app.flags.DEFINE_float("initial_alpha_high", 5e-3, "log_uniform high limit for learning rate")
		tf.app.flags.DEFINE_float("initial_alpha_log_rate", 0.5, "log_uniform interpolate rate for learning rate")
		tf.app.flags.DEFINE_float("gamma", 0.95, "discount factor for rewards")
		tf.app.flags.DEFINE_float("entropy_beta", 0.001, "entropy regularization constant")
		tf.app.flags.DEFINE_integer("max_time_step", 10 * 10**7, "max time steps")
		tf.app.flags.DEFINE_integer("save_interval_step", 100 * 1000, "saving interval steps")
		tf.app.flags.DEFINE_float("grad_norm_clip", 40.0, "gradient norm clipping")

	# For display
	if option_type == 'display':
		tf.app.flags.DEFINE_string("frame_save_dir", "/tmp/unreal_frames", "frame save directory")
		tf.app.flags.DEFINE_boolean("recording", False, "whether to record movie")
		tf.app.flags.DEFINE_boolean("frame_saving", False, "whether to save frames")
	
def get():
	return tf.app.flags.FLAGS
