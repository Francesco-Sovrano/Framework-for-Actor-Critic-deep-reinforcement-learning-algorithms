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

import numpy as np
import scipy
import itertools
from abc import ABC, abstractmethod
from math import floor

# ABSTRACT CLASSES

class StateGenerator(ABC):
	@staticmethod
	def screen_shape():
		return (24,80,1)
		
	def __init__(self):
		self._set_shape()
		self._set_situations_count()
		self.reset()
		
	def reset(self):
		self.need_reset = False

	@abstractmethod
	def _set_shape(self):
		"""The implementing class MUST set the state _shape (should be a tuple)."""
		self._shape = (0, 0, 0) # [heigth, width, channel]
		
	def _set_situations_count(self):
		self._situations = 1
		
	def get_status_size(self):
		return 9
		
	def empty_status(self):
		return np.zeros(self.get_status_size(), dtype=np.float32)

	def compute_state(self, info):
		self.player_position = info.get_player_pos( )
		if info.has_statusbar() and self.player_position != None:
			value, situation = self.build_state(info)
			return { "value" : value, "situation" : situation }
		return { "value" : self.empty_state(), "situation" : 0 }
		
	@abstractmethod
	def build_state(self, info):
		pass
		
	@abstractmethod
	def move_agent_in_all_known_walkable_positions(self, info):
		pass

	def set_positions(self, state, positions, value):
		for pos in positions:
			if pos:
				i, j = pos
				state[i][j] = value
		return state
		
	def set_channel(self, channel, state, positions, value):
		for pos in positions:
			if pos:
				i, j = pos
				state[i][j][channel] = value
		return state
	
	def empty_state(self):
		return np.zeros(self._shape, dtype=np.float32)
		
	@staticmethod
	def environment_tiles_are_in_position_range(info, tiles, position, r):
		x, y = position
		for i in range (-r,r+1):
			a = x + i
			for j in range (-r,r+1):
				b = y + j
				pixel = info.get_environment_tile_at((a, b))
				if pixel in tiles:
					return True
		return False

class ScreenView_StateGenerator(StateGenerator):
	def _set_shape(self):
		self._shape = self.screen_shape()
		
	def move_agent_in_all_known_walkable_positions(self, info):
		player_position = info.get_player_pos()
		if not info.has_statusbar() or player_position is None:
			return None
		result = []
		list_of_walkable_positions = info.get_list_of_walkable_positions()
		px, py = player_position
		for walkable_position in list_of_walkable_positions:
			wx, wy = walkable_position
			state = self.build_state(info)
			state[px][py][0] = ord(info.get_tile_below_player()) # set player position as tile below player
			state[wx][wy][0] = ord('@') # set walkable position as player
			result.append((state,walkable_position))
		return result
		
	def build_state(self, info):
		state = np.array([[[ord(char)] for char in line] for line in info.screen])
		return state, 0
		
class FullView_StateGenerator(StateGenerator): # 1 situation
	def _set_shape(self):
		(screen_x,screen_y,_) = self.screen_shape()
		self._shape = (screen_x-2, screen_y, 1) # [heigth, width, channel]

	def move_agent_in_all_known_walkable_positions(self, info):
		player_position = info.get_player_pos()
		if not info.has_statusbar() or player_position is None:
			return None
		result = []
		list_of_walkable_positions = info.get_list_of_walkable_positions()
		for walkable_position in list_of_walkable_positions:
			self.player_position = walkable_position
			state, _ = self.build_state(info)
			result.append((state,walkable_position))
		return result
		
	def build_state(self, info):
		state = self.empty_state()
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("%"), 4) # stairs
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("|"), 8) # walls
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("-"), 8) # walls
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("+"), 16) # doors
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("#"), 16) # tunnel
		# set it for last otherwise it may be overwritten by other positions!
		state = self.set_channel(0, state, [self.player_position], 2) # rogue (player)
		return state, 0

class C1S3_FullView_StateGenerator(FullView_StateGenerator): # 3 situations		
	def _set_situations_count(self):
		self._situations = 3
		
	def build_state(self, info):
		state, situation = super().build_state(info)
			
		pixel = info.get_tile_below_player()
		if pixel == '#': # tunnel
			situation = 0
		elif pixel == "%": # stairs
			situation = 1
		else:
			situation = 2
		return state, situation

class CroppedView_StateGenerator(StateGenerator): # 6 situations
		
	def move_agent_in_all_known_walkable_positions(self, info):
		player_position = info.get_player_pos()
		if not info.has_statusbar() or player_position is None:
			return []
		result = []
		list_of_walkable_positions = info.get_list_of_walkable_positions()
		for walkable_position in list_of_walkable_positions:
			self.player_position = walkable_position
			state, _ = self.build_state(info)
			result.append((state,walkable_position))
		return result
		
	def _get_relative_coordinates(self, tile_position, centre_position, range):
		i, j = tile_position
		x, y = centre_position
		norm_i = i-x+range[0]//2
		norm_j = j-y+range[1]//2
		return norm_i, norm_j
		
	def is_valid_coordinate(self, i, j):
		return i >= 0 and j >= 0 and i < self._shape[0] and j < self._shape[1]
		
	def set_channel(self, channel, centre_position, state, positions, value):
		for pos in positions:
			if pos:
				i, j = self._get_relative_coordinates(pos, centre_position, self._shape)
				if self.is_valid_coordinate(i, j):
					state[i][j][channel] = value
		return state
		
	def _set_shape(self):
		self._shape = (17, 17, 1) # [heigth, width, channel]
		
	def _set_situations_count(self):
		self._situations = 6
		
	def build_state(self, info):
		state = self.empty_state()
		state = self.set_channel(0, self.player_position, state, info.get_list_of_positions_by_tile("%"), 4) # stairs
		state = self.set_channel(0, self.player_position, state, info.get_list_of_positions_by_tile("|"), 8) # walls
		state = self.set_channel(0, self.player_position, state, info.get_list_of_positions_by_tile("-"), 8) # walls
		state = self.set_channel(0, self.player_position, state, info.get_list_of_positions_by_tile("+"), 16) # doors
		state = self.set_channel(0, self.player_position, state, info.get_list_of_positions_by_tile("#"), 16) # tunnel
			
		pixel = info.get_tile_below_player()
		if pixel == '#': # situation 1
			return state, 0
		if pixel == '+': # situation 2
			return state, 1
		if pixel == "%": # situation 3
			return state, 2
			
		if info.get_tile_count("%") > 0: # situation 4
			return state, 3

		if self.environment_tiles_are_in_position_range(info, "|-", self.player_position, 1): # situation 5
			return state, 4
		return state, 5

class C1S4_CroppedView_StateGenerator(CroppedView_StateGenerator): # 4 situations

	def _set_situations_count(self):
		self._situations = 4
		
	def build_state(self, info):
		state = self.empty_state()
		state = self.set_channel(0, self.player_position, state, info.get_list_of_positions_by_tile("%"), 4) # stairs
		state = self.set_channel(0, self.player_position, state, info.get_list_of_positions_by_tile("|"), 8) # walls
		state = self.set_channel(0, self.player_position, state, info.get_list_of_positions_by_tile("-"), 8) # walls
		state = self.set_channel(0, self.player_position, state, info.get_list_of_positions_by_tile("+"), 16) # doors
		state = self.set_channel(0, self.player_position, state, info.get_list_of_positions_by_tile("#"), 16) # tunnel
			
		pixel = info.get_tile_below_player()
		if pixel == '#': # situation 0
			return state, 0
			
		if info.get_tile_count("%") > 0: # situation 1
			return state, 1

		if self.environment_tiles_are_in_position_range(info, "|-", self.player_position, 1): # situation 2
			return state, 2
		return state, 3

class C2S4_CroppedView_StateGenerator(CroppedView_StateGenerator): # 4 situations
	def _set_shape(self):
		self._shape = (17, 17, 2) # [heigth, width, channel]

	def _set_situations_count(self):
		self._situations = 4
		
	def build_state(self, info):
		state = self.empty_state()
		state = self.set_channel(1, self.player_position, state, info.get_list_of_positions_by_tile("%"), 4) # stairs
		state = self.set_channel(0, self.player_position, state, info.get_list_of_positions_by_tile("|"), 8) # walls
		state = self.set_channel(0, self.player_position, state, info.get_list_of_positions_by_tile("-"), 8) # walls
		state = self.set_channel(0, self.player_position, state, info.get_list_of_positions_by_tile("+"), 16) # doors
		state = self.set_channel(0, self.player_position, state, info.get_list_of_positions_by_tile("#"), 16) # tunnel
			
		pixel = info.get_tile_below_player()
		if pixel == '#': # situation 0
			return state, 0
			
		if info.get_tile_count("%") > 0: # situation 1
			return state, 1

		if self.environment_tiles_are_in_position_range(info, "|-", self.player_position, 1): # situation 2
			return state, 2
		return state, 3
		
class Complete_CroppedView_StateGenerator(CroppedView_StateGenerator):
	def _set_shape(self):
		self._shape = (17, 17, 6) # [heigth, width, channel]
		
	def compute_state(self, info):
		self.player_position = info.get_player_pos()
		if info.has_statusbar() and self.player_position != None:
			value, status = self.build_state(info)
			return { "value" : value, "status" : status }
		return { "value" : self.empty_state(), "status" : self.empty_status() }
		
	def build_state(self, info):
		map = self.empty_state()
		map = self.set_channel(3, self.player_position, map, info.get_list_of_positions_by_tile("%"), 1) # stairs
		map = self.set_channel(2, self.player_position, map, info.get_list_of_positions_by_tile("|"), 1) # walls
		map = self.set_channel(2, self.player_position, map, info.get_list_of_positions_by_tile("-"), 1) # walls
		map = self.set_channel(1, self.player_position, map, info.get_list_of_positions_by_tile("+"), 1) # doors
		map = self.set_channel(0, self.player_position, map, info.get_list_of_positions_by_tile("#"), 1) # tunnel
		for i,item in enumerate(info.pixel["items"]): # items
			map = self.set_channel(5, self.player_position, map, info.pixel["items"][item], i+1)
		for i,monster in enumerate(info.pixel["monsters"]): # monsters
			map = self.set_channel(4, self.player_position, map, info.pixel["monsters"][monster], i+1)
		
		status = self.empty_status()
		status[0] = info.statusbar["gold"]
		status[1] = info.statusbar["current_hp"]
		status[2] = info.statusbar["max_hp"]
		status[3] = info.statusbar["current_strength"]
		status[4] = info.statusbar["max_strength"]
		status[5] = info.statusbar["armor"]
		status[6] = info.statusbar["tot_exp"]
		status[7] = info.statusbar["exp_level"]
		status[8] = info.statusbar["command_count"]
		return map, status
		
class Complete_FullView_StateGenerator(FullView_StateGenerator):
	def _set_shape(self):
		(screen_x,screen_y,_) = self.screen_shape()
		self._shape = (screen_x-2, screen_y, 7) # [heigth, width, channel]
		
	def compute_state(self, info):
		self.player_position = info.get_player_pos()
		if info.has_statusbar() and self.player_position != None:
			value, status = self.build_state(info)
			return { "value" : value, "status" : status }
		return { "value" : self.empty_state(), "status" : self.empty_status() }
		
	def build_state(self, info):
		map = self.empty_state()
		map = self.set_channel(3, map, info.get_list_of_positions_by_tile("%"), 1) # stairs
		map = self.set_channel(2, map, info.get_list_of_positions_by_tile("|"), 1) # walls
		map = self.set_channel(2, map, info.get_list_of_positions_by_tile("-"), 1) # walls
		map = self.set_channel(1, map, info.get_list_of_positions_by_tile("+"), 1) # doors
		map = self.set_channel(0, map, info.get_list_of_positions_by_tile("#"), 1) # tunnel
		for i,item in enumerate(info.pixel["items"]): # items
			map = self.set_channel(5, map, info.pixel["items"][item], i+1)
		for i,monster in enumerate(info.pixel["monsters"]): # monsters
			map = self.set_channel(4, map, info.pixel["monsters"][monster], i+1)
		map = self.set_channel(6, map, [self.player_position], 1) # rogue (player)
		
		status = self.empty_status()
		status[0] = info.statusbar["gold"]
		status[1] = info.statusbar["current_hp"]
		status[2] = info.statusbar["max_hp"]
		status[3] = info.statusbar["current_strength"]
		status[4] = info.statusbar["max_strength"]
		status[5] = info.statusbar["armor"]
		status[6] = info.statusbar["tot_exp"]
		status[7] = info.statusbar["exp_level"]
		status[8] = info.statusbar["command_count"]
		return map, status