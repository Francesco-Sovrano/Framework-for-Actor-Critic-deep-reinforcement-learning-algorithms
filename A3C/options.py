# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

options_built = False
def build():
	tf.app.flags.DEFINE_boolean("use_gpu", False, "whether to use the GPU")
	tf.app.flags.DEFINE_integer("max_time_step", 2*10**8, "max time steps")
# Environment
	# tf.app.flags.DEFINE_string("env_type", "MontezumaRevenge-ram-v0", "environment types: rogue, or environments from https://gym.openai.com/envs")
	tf.app.flags.DEFINE_string("env_type", "rogue", "environment types: rogue, or environments from https://gym.openai.com/envs")
# Gradient optimization parameters
	tf.app.flags.DEFINE_string("network", "BaseAC", "neural network: BaseAC") # default is Adam, for vanilla A3C is RMSProp
	tf.app.flags.DEFINE_string("optimizer", "Adam", "gradient optimizer: Adadelta, AdagradDA, Adagrad, Adam, Ftrl, GradientDescent, Momentum, ProximalAdagrad, ProximalGradientDescent, RMSProp") # default is Adam, for vanilla A3C is RMSProp
	tf.app.flags.DEFINE_float("grad_norm_clip", 0, "gradient norm clipping (0 for none)") # default is 40.0, for openAI is 0.5
	tf.app.flags.DEFINE_string("policy_loss", "PPO", "policy loss function: vanilla, PPO, averagePPO, openaiPPO") # usually averagePPO works with GAE
	tf.app.flags.DEFINE_string("value_loss", "vanilla", "value loss function: vanilla, PVO, averagePVO") # usually averagePVO works with GAE
# Partitioner parameters
	# Partition count > 0 reduces algorithm speed, because also a partitioner is trained
	tf.app.flags.DEFINE_integer("partition_count", 5, "Number of partitions of the input space. Set to 1 for no partitions.")
	# Partitioner granularity > 0 increases algorithm speed when partition_count > 0
	tf.app.flags.DEFINE_integer("partitioner_granularity", 8, "Number of steps after which to run the partitioner.")
	
	tf.app.flags.DEFINE_string("partitioner_type", "ReinforcementLearning", "Partitioner types: ReinforcementLearning, KMeans")
	# Flags for partitioner_type == KMeans
	tf.app.flags.DEFINE_integer("partitioner_training_set_size", 10**5, "Should be a number greater than 0")
	# Flags for partitioner_type == ReinforcementLearning
	tf.app.flags.DEFINE_float("partitioner_learning_factor", 2, "Should be a number greater than 0. Usually the partitioner has an higher learning rate than the others. This factor is used to change the initial learning rate of the partitioner only.") # default is 2.0
	tf.app.flags.DEFINE_string("partitioner_optimizer", "ProximalAdagrad", "gradient optimizer: Adadelta, AdagradDA, Adagrad, Adam, Ftrl, GradientDescent, Momentum, ProximalAdagrad, ProximalGradientDescent, RMSProp") # default is ProximalAdagrad
# Loss clip range
	tf.app.flags.DEFINE_float("clip", 0.2, "PPO/PVO initial clip range") # default is 0.2, for openAI is 0.1
	tf.app.flags.DEFINE_boolean("clip_decay", False, "Whether to decay the clip range")
	tf.app.flags.DEFINE_string("clip_annealing_function", "exponential_decay", "annealing function: exponential_decay, inverse_time_decay, natural_exp_decay") # default is inverse_time_decay
	tf.app.flags.DEFINE_integer("clip_decay_steps", 10**5, "decay clip every x steps") # default is 1
	tf.app.flags.DEFINE_float("clip_decay_rate", 0.96, "decay rate") # default is 0.5
# Learning rate
	tf.app.flags.DEFINE_float("alpha", 3.5e-4, "initial learning rate") # default is 7.0e-4, for openAI is 2.5e-4
	tf.app.flags.DEFINE_boolean("alpha_decay", False, "whether to decay the learning rate") # default is False
	tf.app.flags.DEFINE_string("alpha_annealing_function", "exponential_decay", "annealing function: exponential_decay, inverse_time_decay, natural_exp_decay")
	tf.app.flags.DEFINE_integer("alpha_decay_steps", 10**5, "decay alpha every x steps")
	tf.app.flags.DEFINE_float("alpha_decay_rate", 0.96, "decay rate")
# Last Action-Reward: Jaderberg, Max, et al. "Reinforcement learning with unsupervised auxiliary tasks." arXiv preprint arXiv:1611.05397 (2016).
	tf.app.flags.DEFINE_boolean("concat_last_action_reward", True, "Whether to concatenate the last action-reward vector in the network.")
# Reward Prediction: Jaderberg, Max, et al. "Reinforcement learning with unsupervised auxiliary tasks." arXiv preprint arXiv:1611.05397 (2016).
	tf.app.flags.DEFINE_boolean("predict_reward", True, "Whether to predict rewards. This is useful with sparse rewards.")
	tf.app.flags.DEFINE_integer("reward_prediction_buffer_size", 128, "Maximum number of batches stored in the reward prediction buffer")
# Experience Replay
	# Replay ratio > 0 increases off-policyness
	tf.app.flags.DEFINE_float("replay_ratio", 1, "Mean number of experience replays per batch. Lambda parameter of a Poisson distribution. When replay_ratio is 0, then experience replay is de-activated.") # for A3C is 0, for ACER default is 4
	tf.app.flags.DEFINE_integer("replay_step", 10**3, "Start replaying when global step is greater than replay_step.")
	tf.app.flags.DEFINE_boolean("replay_value", False, "Whether to recompute values, advantages and discounted cumulative rewards") # default is True
	tf.app.flags.DEFINE_integer("replay_buffer_size", 64, "Maximum number of batches stored in the experience replay buffer")
	tf.app.flags.DEFINE_integer("replay_start", 1, "Should be greater than 0 and lower than replay_buffer_size. Train on x batches before using experience replay") # default is 5000
	tf.app.flags.DEFINE_boolean("save_only_batches_with_reward", True, "Save in the replay buffer only those batches with total reward different from 0") # default is True
# Reward clip
	tf.app.flags.DEFINE_boolean("clip_reward", False, "Whether to clip the reward between min_reward and max_reward") # default is False
	tf.app.flags.DEFINE_float("min_reward", 0, "Minimum reward for clipping") # default is -1
	tf.app.flags.DEFINE_float("max_reward", 1, "Maximum reward for clipping") # default is 1
# Actor-Critic parameters
	# Learning rate for Critic is half of Actor's, so multiply by 0.5 (default)
	tf.app.flags.DEFINE_float("value_coefficient", 0.5, "value coefficient for tuning Critic learning rate") # default is 0.5, for openAI is 0.25
	tf.app.flags.DEFINE_float("entropy_beta", 0.001, "entropy regularization constant") # default is 0.001, for openAI is 0.01
	tf.app.flags.DEFINE_integer("parallel_size", 4, "parallel thread size")
	tf.app.flags.DEFINE_integer("max_batch_size", 8, "maximum batch size") # default is 60, for openAI is 128
	# Taking gamma < 1 introduces bias into the policy gradient estimate, regardless of the value function’s accuracy.
	tf.app.flags.DEFINE_float("gamma", 0.99, "discount factor for rewards") # default is 0.95, for openAI is 0.99
# Generalized Advantage Estimation
	tf.app.flags.DEFINE_boolean("use_GAE", True, "whether to use Generalized Advantage Estimation (default in openAI's PPO implementation)") # Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).
	# Taking lambda < 1 introduces bias only when the value function is inaccurate
	tf.app.flags.DEFINE_float("lambd", 0.95, "generalized advantage estimator decay parameter") # default is 0.95
# Log
	tf.app.flags.DEFINE_integer("save_interval_step", 10**6, "saving interval steps")
	tf.app.flags.DEFINE_integer("match_count_for_evaluation", 200, "number of matches used for evaluation scores")
	tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint", "checkpoint directory")
	tf.app.flags.DEFINE_string("event_dir", "./events", "events directory")
	tf.app.flags.DEFINE_string("log_dir", "./log", "events directory")
	tf.app.flags.DEFINE_boolean("show_best_episodes", True, "whether to save best matches")
	tf.app.flags.DEFINE_boolean("show_all_episodes", False, "whether to save all the matches")
	# save_episode_screen = True might slow down the algorithm
	tf.app.flags.DEFINE_boolean("save_episode_screen", False, "whether to save episode screens")
	# save_episode_heatmap = True slows down the algorithm
	tf.app.flags.DEFINE_boolean("save_episode_heatmap", False, "whether to save episode heatmap")
	# save_episode_gif = True slows down the algorithm
	tf.app.flags.DEFINE_boolean("save_episode_gif", True, "whether to save episode gif")
	tf.app.flags.DEFINE_float("gif_speed", 0.1, "gif speed in seconds")
# Plot
	tf.app.flags.DEFINE_boolean("compute_plot_when_saving", True, "Whether to compute the plot when saving checkpoints")
	tf.app.flags.DEFINE_integer("max_plot_size", 1000, "Maximum number of points in the plot. The smaller it is, the less RAM is required. If the log file has more than max_plot_size points, then max_plot_size means of slices are used instead.")
# Rogue stuff
	tf.app.flags.DEFINE_string("state_generator", "Channel6_Complete_CroppedView_StateGenerator", "the state generator must be a classname from rogueinabox/states.py")
	# tf.app.flags.DEFINE_string("reward_generator", "Improved_ENSS_RewardGenerator", "the reward generator must be a classname from rogueinabox/rewards.py")
	tf.app.flags.DEFINE_string("reward_generator", "Stair_RewardGenerator", "the reward generator must be a classname from rogueinabox/rewards.py")
	tf.app.flags.DEFINE_integer("steps_per_episode", 500, "number of maximum actions execution per episode")
	tf.app.flags.DEFINE_string("env_path", "./Rogue/rogue5.4.4-ant-r1.1.4_monsters/rogue", "the path where to find the game")
	tf.app.flags.DEFINE_string("rogueinabox_path", "./Rogue", "where to find the package") # to remove!
	
	global options_built
	options_built = True
	
def get():
	global options_built
	if not options_built:
		build()
	return tf.app.flags.FLAGS