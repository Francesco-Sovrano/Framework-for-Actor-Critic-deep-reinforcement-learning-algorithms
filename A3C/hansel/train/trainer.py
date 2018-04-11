# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging
import numpy as np
import time

from environment.environment import Environment
from model.model import MultiAgentModel

# get command line args
import options
flags = options.get()

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000


class Trainer(object):
	def __init__(self, thread_index, global_network, initial_learning_rate, learning_rate_input, grad_applier, env_type, entropy_beta, local_t_max, gamma, max_global_time_step, device):
		self.stats = {}
		self.thread_index = thread_index
		self.global_network = global_network
		self.grad_applier = grad_applier
	#logs
		formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
		
		self.info_logger = logging.getLogger('info_' + str(thread_index))
		hdlr = logging.FileHandler(flags.log_dir + '/performance/info_' + str(thread_index) + '.log')
		hdlr.setFormatter(formatter)
		self.info_logger.addHandler(hdlr) 
		self.info_logger.setLevel(logging.DEBUG)
		
		self.reward_logger = logging.getLogger('reward_' + str(thread_index))
		hdlr = logging.FileHandler(flags.log_dir + '/performance/reward_' + str(thread_index) + '.log')
		hdlr.setFormatter(formatter)
		self.reward_logger.addHandler(hdlr) 
		self.reward_logger.setLevel(logging.DEBUG)

		self.max_reward = float("-inf")
	#trainer
		self.learning_rate_input = learning_rate_input
		self.env_type = env_type
		self.local_t_max = local_t_max
		self.gamma = gamma
		self.environment = Environment.create_environment(self.env_type, self.thread_index)
		self.action_size = self.environment.get_action_size()
		state_shape = self.environment.get_state_shape()
		agents_count = self.environment.get_situations_count()
		self.max_global_time_step = max_global_time_step
		self.entropy_beta = entropy_beta
		self.device = device
	# build network
		self.local_network = MultiAgentModel(self.thread_index, state_shape, agents_count, self.action_size, self.entropy_beta, self.device)
		self.apply_gradients = []
		self.sync = []
		for i in range(self.local_network.agent_count):
			local_agent = self.local_network.get_agent(i)
			global_agent = self.global_network.get_agent(i)
			local_agent.prepare_loss()
			self.apply_gradients.append( self.grad_applier.minimize_local(local_agent.total_loss, global_agent.get_vars(), local_agent.get_vars()) )
			self.sync.append( local_agent.sync_from(global_agent) )
		self.local_t = 0
		self.initial_learning_rate = initial_learning_rate
	# For log output
		self.prev_local_t = 0

	def prepare(self):
		self.episode_reward = 0
		self.episode_steps = 0
		self.environment.reset()
		self.local_network.reset()

	def stop(self):
		self.environment.stop()
		
	def _anneal_learning_rate(self, global_time_step):
		learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
		if learning_rate < 0.0:
			learning_rate = 0.0
		return learning_rate

	def choose_action(self, pi_values):
		return np.random.choice(range(len(pi_values)), p=pi_values)

	def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
		summary_str = sess.run(summary_op, feed_dict={
			score_input: score
		})
		summary_writer.add_summary(summary_str, global_t)
		summary_writer.flush()
	
	def set_start_time(self, start_time):
		self.start_time = start_time

	def _print_log(self, global_t):
		if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
			self.prev_local_t += PERFORMANCE_LOG_INTERVAL
			elapsed_time = time.time() - self.start_time
			steps_per_sec = global_t / elapsed_time
			print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format( global_t, elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))
		
	# [Base A3C]
	def _process_base(self, sess, global_t, summary_writer, summary_op, score_input):
		batch = {}
		batch["states"] = []
		batch["action_maps"] = []
		batch["action_rewards"] = []
		batch["rewards"] = []
		batch["adversarial_reward"] = []
		batch["start_lstm_state"] = []
		for i in range(self.local_network.agent_count):
			for key in batch:
				batch[key].append(collections.deque())
			batch["start_lstm_state"][i] = self.local_network.get_agent(i).base_lstm_state_out
			
		states = collections.deque()
		action_rewards = collections.deque()
		actions = collections.deque()
		rewards = collections.deque()
		values = collections.deque()
		terminal_end = False
		new_state = None
		
		# t_max times loop
		for _ in range(self.local_t_max):
			# Prepare last action reward
			prev_state = self.environment.last_state
			last_action = self.environment.last_action
			last_reward = self.environment.last_reward
			last_action_reward = self.local_network.concat_action_and_reward(last_action, last_reward)
			
			agent = self.local_network.get_agent(prev_state["situation"])
			pi_, value_ = agent.run_policy_and_value(sess, prev_state["value"], last_action_reward)
			action = self.choose_action(pi_)
			
			states.appendleft(prev_state)
			action_rewards.appendleft(last_action_reward)
			actions.appendleft(action)
			values.appendleft(value_)
			
			if (self.local_t % LOG_INTERVAL == 0):
				self.info_logger.info(
					" actions={}".format(pi_) +
					" value={}".format(value_) +
					" agent={}".format(prev_state["situation"])
				)

			# Process game
			new_state, reward, win, lose = self.environment.process(action)

			# Store to experience
			self.episode_reward += reward
			self.episode_steps += 1

			rewards.appendleft(reward)
			self.local_t += 1

			if win or lose:
				self.stats = self.environment.get_statistics()
				log_str = ""
				for key in self.stats:
					log_str += " " + key + "=" + str(self.stats[key])
				self.reward_logger.info(
					" score={}".format(self.episode_reward) +
					" steps={}".format(self.episode_steps) + 
					log_str
				)
				# print("thread" + str(self.thread_index) + " score=" + str(self.episode_reward))
				if flags.show_all_screenshots:
					self.environment.print_display(global_t, self.episode_reward)
				elif flags.show_best_screenshots:
					if self.episode_reward >= self.max_reward:
						self.max_reward = self.episode_reward
						self.environment.print_display(global_t, self.episode_reward)
				elif flags.show_all_screenshots:
					self.max_reward = self.episode_reward
					self.environment.print_display(global_t, self.episode_reward)
				self._record_score(sess, summary_writer, summary_op, score_input, self.episode_reward, global_t)
					
				self.prepare()
				break

		# If we episode was not done we bootstrap the value from the last state
		R = 0.0
		if not (win or lose):
			agent = self.local_network.get_agent(new_state["situation"])
			R = agent.run_value(sess, new_state["value"], self.local_network.concat_action_and_reward(actions[0], rewards[0]))
			
		for(action, reward, state, value, action_reward) in zip(actions, rewards, states, values, action_rewards):
			R = reward + self.gamma * R
			adversarial_reward = R - value
			action_map = np.zeros([self.action_size])
			action_map[action] = 1.0
			agent_id = state["situation"]
			batch["states"][agent_id].appendleft( state["value"] )
			batch["action_maps"][agent_id].appendleft( action_map )
			batch["rewards"][agent_id].appendleft( R )
			batch["action_rewards"][agent_id].appendleft( action_reward )
			batch["adversarial_reward"][agent_id].appendleft( adversarial_reward )
		
		return batch
	
	def process(self, sess, global_t, summary_writer, summary_op, score_input):
		start_local_t = self.local_t
		cur_learning_rate = self._anneal_learning_rate(global_t)

		# Copy weights from shared to local
		for i in range(self.local_network.agent_count):
			sess.run( self.sync[i] )

		try:
			# Build feed dictionary
			batch_base = self._process_base(sess, global_t, summary_writer, summary_op, score_input)				
			# Pupulate the feed dictionary
			for i in range(self.local_network.agent_count):
				if len(batch_base["states"][i]) > 0:
					agent = self.local_network.get_agent(i)
					feed_dict = {
						self.learning_rate_input: cur_learning_rate,
						agent.base_input: batch_base["states"][i],
						agent.base_last_action_reward_input: batch_base["action_rewards"][i],
						agent.base_a: batch_base["action_maps"][i],
						agent.base_adv: batch_base["adversarial_reward"][i],
						agent.base_r: batch_base["rewards"][i],
						agent.base_initial_lstm_state: batch_base["start_lstm_state"][i],
					}
					
					# Calculate gradients and copy them to global network.
					sess.run( self.apply_gradients[i], feed_dict )
					self._print_log(global_t)
			
			# Return advanced local step size
			return self.local_t - start_local_t
		except:
			self.prepare()
			return 0
