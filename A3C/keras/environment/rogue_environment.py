# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# get command line args
import options
flags = options.get()

from environment import environment
from .situations import rogue_situations
from .rewards import rogue_rewards
from .states import rogue_states
from roguelib_module.rogueinabox import RogueBox


class RogueEnvironment(environment.Environment):
	def get_action_size(self):
		return len(self.real_actions)

	def get_state_shape(self):
		return self.game.state_generator.get_shape()

	def __init__(self, thread_index):
		super().__init__()
		self.thread_index = thread_index
		self.real_actions = RogueBox.get_actions()
		self.game = RogueBox(game_exe_path=flags.env_path or None,
							 use_monsters=flags.use_monsters,
							 max_step_count=flags.steps_per_episode,
							 episodes_for_evaluation=flags.match_count_for_evaluation,
							 state_generator=self._instantiate_from_module(rogue_states, flags.state_generator),
							 reward_generator=self._instantiate_from_module(rogue_rewards, flags.reward_generator),
							 refresh_after_commands=False,
							 move_rogue=True)

	def _create_situation_generator(self):
		return self._instantiate_from_module(rogue_situations, flags.situation_generator, on_exc_return_name=False)

	def reset(self):
		super().reset()
		if flags.show_best_screenshots or flags.show_all_screenshots:
			self.screenshots = list()
			self.screeninfo = list()
		self.last_action = self.real_actions.index('>')
		self.last_reward, self.last_state, _, _ = self.game.reset()
		self.last_situation = self._situation_generator.compute_situation(self.game.frame_history)

	def stop(self):
		self.game.stop()

	def _process_frame(self, action):
		return self.game.send_command(action)

	def get_screen(self):
		return self.game.get_screen()

	def get_statistics(self):
		return self.game.evaluator.statistics()

	def _save_display(self):
		self.screenshots.append(self.game.get_screen())
		last_frame = self.game.get_last_frame()
		self.screeninfo.append("reward: {2}, passages: {0}, doors: {1}, below_player: ".format(
			last_frame.get_tile_count("#"),
			last_frame.get_tile_count("+"),
			self.game.reward
		) + last_frame.get_tile_below_player() + "\n")

	def print_display(self, step, reward):
		file = open(flags.log_dir + '/screenshots/reward(' + str(reward) + ')_step(' + str(step) + ')_thread(' + str(self.thread_index) + ').log', "w")
		for i in range(len(self.screenshots)):
			file.write(self.screeninfo[i])
			screen = self.screenshots[i]
			for line in screen:
				file.write(str(line) + '\n')
		file.close()

	def process(self, action):
		real_action = self.real_actions[action]
		reward, new_state, win, lose = self._process_frame(real_action)

		self.last_state = new_state
		self.last_action = action
		self.last_reward = reward
		self.last_situation = self._situation_generator.compute_situation(self.game.frame_history)

		if flags.show_best_screenshots or flags.show_all_screenshots:
			self._save_display()

		return new_state, reward, win, lose
