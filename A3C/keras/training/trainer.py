# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
	def __init__(self, thread_index, global_weights, local_network, env_type, local_t_max, gamma, max_global_time_step, device):
		self.stats = {}
		self.thread_index = thread_index
		self.global_weights = global_weights
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
		self.env_type = env_type
		self.local_t_max = local_t_max
		self.gamma = gamma
		self.environment = Environment.create_environment(self.env_type, self.thread_index)
		self.action_size = self.environment.get_action_size()
		self.max_global_time_step = max_global_time_step
		self.local_t = 0
		self.local_network = local_network
		""":type : MultiAgentModel"""
	# For log output
		self.prev_local_t = 0

	def prepare(self):
		self.episode_reward = 0
		self.episode_steps = 0
		self.environment.reset()
		self.local_network.reset()

	def stop(self):
		self.environment.stop()

	def choose_action(self, pi_values):
		return np.random.choice(range(len(pi_values)), p=pi_values)
	
	def set_start_time(self, start_time):
		self.start_time = start_time

	def _print_log(self, global_t):
		if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
			self.prev_local_t += PERFORMANCE_LOG_INTERVAL
			elapsed_time = time.time() - self.start_time
			steps_per_sec = global_t / elapsed_time
			print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format( global_t, elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))
		
	# [Base A3C]
	def _process_base(self, global_t):
		batch = {}
		batch["states"] = []
		batch["action_maps"] = []
		batch["action_rewards"] = []
		batch["rewards"] = []
		batch["adversarial_reward"] = []
		batch["start_lstm_state"] = []
		for i in range(self.local_network.agent_count):
			for key in batch:
				batch[key].append([])
			batch["start_lstm_state"][i] = self.local_network.get_agent(i).get_state()
			
		states = []
		action_rewards = []
		actions = []
		rewards = []
		values = []
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
			pi_, [value_] = agent.run_policy_and_value(prev_state["value"], last_action_reward)
			action = self.choose_action(pi_)
			
			states.append( prev_state )
			action_rewards.append(last_action_reward)
			actions.append(action)
			values.append(value_)
			
			if (self.local_t % LOG_INTERVAL == 0):
				self.info_logger.info(
					" actions={}".format(pi_) +
					" value={}".format(value_) +
					" agent={}".format(prev_state["situation"])
				)

			# Process game
			new_state, reward, win, lose = self.environment.process(action)
			terminal = win or lose

			# Store to experience
			self.episode_reward += reward
			self.episode_steps += 1

			rewards.append( reward )
			self.local_t += 1

			if terminal:
				terminal_end = True
				
				self.stats = self.environment.game.evaluator.statistics( flags.match_count_for_evaluation )
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
					
				self.prepare()
				break

		actions.reverse()
		states.reverse()
		rewards.reverse()
		values.reverse()
		action_rewards.reverse()

		# If we episode was not done we bootstrap the value from the last state
		R = 0.0
		if not terminal_end:
			agent = self.local_network.get_agent(new_state["situation"])
			[R] = agent.run_value(new_state["value"], self.local_network.concat_action_and_reward(actions[0], rewards[0]))
			
		for(action, reward, state, value, action_reward) in zip(actions, rewards, states, values, action_rewards):
			R = reward + self.gamma * R
			adversarial_reward = R - value
			action_map = np.zeros([self.action_size])
			action_map[action] = 1.0
			agent_id = state["situation"]
			batch["states"][agent_id].append( state["value"] )
			batch["action_maps"][agent_id].append( action_map )
			batch["rewards"][agent_id].append( R )
			batch["action_rewards"][agent_id].append( action_reward )
			batch["adversarial_reward"][agent_id].append( adversarial_reward )
		
		for i in range(self.local_network.agent_count):		
			batch["states"][i].reverse()
			batch["action_maps"][i].reverse()
			batch["rewards"][i].reverse()
			batch["action_rewards"][i].reverse()
			batch["adversarial_reward"][i].reverse()
		
		return batch
	
	def process(self, global_t):
		start_local_t = self.local_t

		# Copy weights from shared to local
		self.local_network.set_weights(self.global_weights)

		# Create batch by playing
		try:
			batch_base = self._process_base(global_t)
		except:
			self.prepare()
			return 0
			
		# Train networks for each situation on the batch
		for i in range(self.local_network.agent_count):
			if len(batch_base["states"][i]) > 0:

				agent = self.local_network.get_agent(i)
				agent.set_state(batch_base["start_lstm_state"][i])

				old_w = agent.get_weights()

				base_input = batch_base["states"][i]
				base_last_action_reward_input = batch_base["action_rewards"][i]
				base_a = batch_base["action_maps"][i]
				base_adv = batch_base["adversarial_reward"][i]
				base_r = batch_base["rewards"][i]

				agent.fit(base_input, base_last_action_reward_input, base_a, base_adv, base_r)

				# calculate weights updates
				new_w = agent.get_weights()
				updates = [new - old for new, old in zip(new_w, old_w)]

				# apply updates to the global network
				for u_i, upd in enumerate(updates):
					self.global_weights[i][u_i] += upd

				self._print_log(global_t)
		
		# Return advanced local step size
		diff_local_t = self.local_t - start_local_t
		return diff_local_t
