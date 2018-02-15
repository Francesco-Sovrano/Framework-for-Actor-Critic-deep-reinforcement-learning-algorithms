#Copyright (C) 2017 Andrea Asperti, Carlo De Pieri, Gianmaria Pedrini, Francesco Sovrano
#
#This file is part of Rogueinabox.
#
#Rogueinabox is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#Rogueinabox is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import time
import os
import fcntl
import pty
import signal
import shlex
import pyte
import re
import numpy as np
import itertools
import scipy
import copy

import rogueinabox.rewards as rewards
import rogueinabox.states as states
from rogueinabox.parser import RogueParser
from rogueinabox.evaluator import RogueEvaluator

class Terminal:
	def __init__(self, columns, lines):
		self.screen = pyte.DiffScreen(columns, lines)
		self.stream = pyte.ByteStream()
		self.stream.attach(self.screen)

	def feed(self, data):
		self.stream.feed(data)

	def read(self):
		return self.screen.display

def open_terminal(command="bash", columns=80, lines=24):
	p_pid, master_fd = pty.fork()
	if p_pid == 0:  # Child.
		path, *args = shlex.split(command)
		args = [path] + args
		env = dict(TERM="linux", LC_ALL="en_GB.UTF-8",
				   COLUMNS=str(columns), LINES=str(lines))
		try:
			os.execvpe(path, args, env)
		except FileNotFoundError:
			print("Could not find the executable in %s. Press any key to exit." % path)
			exit()

	# set non blocking read
	flag = fcntl.fcntl(master_fd, fcntl.F_GETFD)
	fcntl.fcntl(master_fd, fcntl.F_SETFL, flag | os.O_NONBLOCK)
	# File-like object for I/O with the child process aka command.
	p_out = os.fdopen(master_fd, "w+b", 0)
	return Terminal(columns, lines), p_pid, p_out

class RogueBox:
	@staticmethod
	def get_actions():
		"""return the list of actions"""
		# h, j, k, l: ortogonal moves
		# y, u, b, n: diagonal moves
		# >: go downstairs
		# return ['h', 'j', 'k', 'l', '>', 'y', 'u', 'b', 'n']
		return ['h', 'j', 'k', 'l', '>']
	
	"""Start a rogue game and expose interface to communicate with it"""
	def __init__(self, game_exe_path, state_generator, reward_generator, max_step_count):
		self.rogue_path = game_exe_path
		self.parser = RogueParser()
		self.evaluator = RogueEvaluator()
		self.max_step_count = max_step_count
		if self.max_step_count <= 0:
			self.max_step_count = 1
		# initialize environment
		self.reward_generator = getattr(rewards, reward_generator)()
		self.state_generator = getattr(states, state_generator)()
		
	def _start(self):
		self.step_count = 0
		self.episode_reward = 0
		self.frame_info = []
		
		self.parser.reset()
		self.reward_generator.reset()
		self.state_generator.reset()
		self.terminal, self.pid, self.pipe = open_terminal(command=self.rogue_path)
		if not self.is_running(): # check whether game is running
			print("Could not find the executable in %s." % self.rogue_path)
			exit()
		
		# wait until the rogue spawns
		self.screen = self.get_empty_screen()
		self._update_screen()
		while self.game_over(self.screen):
			self._update_screen()
		
		# we move the rogue (up, down, left, right) to be able to know what is the tile below it
		actions = RogueBox.get_actions()
		for i in range(3):
			self.send_command(actions[i])
		return 3, self.send_command(actions[3])

	def reset(self):
		"""kill and restart the rogue process"""
		self.stop()
		return self._start()
			
	def stop(self):
		"""kill and restart the rogue process"""
		if self.is_running():
			self.pipe.close()
			os.kill(self.pid, signal.SIGTERM)
			# wait the process so it doesnt became a zombie
			os.waitpid(self.pid, 0)
		
	def _update_screen(self):
		"""update the virtual screen and the class variable"""
		update = self.pipe.read(65536)
		if update:
			self.terminal.feed(update)
			self.screen = self.terminal.read()
			
	def get_empty_screen(self):
		screen = list()
		for row in range(24):
			value = ""
			for col in range(80):
				value += " "
			screen.append(value)
		return screen

	def print_screen(self):
		"""print the current screen"""
		print(*self.screen, sep='\n')

	def get_screen(self):
		"""return the screen as a list of strings.
		can be treated like a 24x80 matrix of characters (screen[17][42])"""
		return self.screen

	def get_screen_string(self):
		"""return the screen as a single string with \n at EOL"""
		out = ""
		for line in self.screen:
			out += line
			out += '\n'
		return out

	def game_over(self, screen):
		"""check if we are at the game over screen (tombstone)"""
		return not ('Hp:' in screen[-1])
		
	def is_running(self):
		"""check if the rogue process exited"""
		try:
			pid, status = os.waitpid(self.pid, os.WNOHANG)
		except:
			return False
		if pid == 0:
			return True
		else:
			return False

	def compute_state(self, new_info):
		"""return a numpy array representation of the current state using the function specified during init"""
		return self.state_generator.compute_state(new_info)

	def compute_reward(self, frame_history):
		"""return the reward for a state transition using the function specified during init"""
		return self.reward_generator.compute_reward(frame_history)

	def _dismiss_message(self):
		"""dismiss a rogue status message.
		call it once, because it will call itself again until
		all messages are dismissed """
		messagebar = self.screen[0]
		if "ore--" in messagebar:
			# press space
			self.pipe.write(' '.encode())
		elif "all it" in messagebar:
			# press esc
			self.pipe.write('\e'.encode())

	def _need_to_dismiss(self):
		"""check if there are status messages that need to be dismissed"""
		messagebar = self.screen[0]
		if "all it" in messagebar or "ore--" in messagebar:
			return True
		else:
			return False

	def quit_the_game(self):
		"""Send the keystroke needed to quit the game."""
		self.pipe.write('Q'.encode())
		self.pipe.write('y'.encode())
		self.pipe.write('\n'.encode())
		
	# interact with rogue methods		
	def send_command(self, command):
		"""send a command to rogue"""
		old_screen = self.screen
		self.pipe.write(command.encode())
		new_screen = old_screen
		while old_screen[-1] == new_screen[-1]: # after a command execution, the new screen is always different from the old one
			# print (self.screen[-1])
			self._update_screen()
			while self._need_to_dismiss(): # will dismiss all upcoming messages
				self._dismiss_message()
				self._update_screen()
			new_screen = self.screen
		
		lose = self.game_over(new_screen)
		if not lose:
			if not self.frame_info:
				self.frame_info.append( self.parser.parse_screen( old_screen ) )
			self.frame_info.append( self.parser.parse_screen( new_screen ) )
			self.reward = self.compute_reward( self.frame_info ) # use last frame history
			self.state = self.compute_state( self.frame_info[-1] ) # use last frame info
			
			self.step_count += 1
			self.episode_reward += self.reward
			lose = self.step_count > self.max_step_count or self.state_generator.need_reset
		
		win = self.reward_generator.goal_achieved
		if win or lose:
			self.evaluator.add( info = self.frame_info[-1], reward = self.episode_reward, has_won = win, step = self.step_count )
		return self.reward, self.state, win, lose