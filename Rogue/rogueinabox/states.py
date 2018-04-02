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
	def __init__(self):
		self._set_shape()
		self._set_situations_count()
		self.reset()
		
	def reset(self):
		self.need_reset = False

	def _set_shape(self):
		"""The implementing class MUST set the state _shape (should be a tuple)."""
		self._shape = (0, 0, 0) # [heigth, width, channel]
		
	def _set_situations_count(self):
		self._situations = 1 # situations are for multi-agent models

	def compute_state(self, info):
		"""Should compute the state and return it."""
		if info.has_statusbar():
			value, situation = self.build_state(info)
			return { "value" : value, "situation" : situation }
		return { "value" : self.empty_state(), "situation" : 0 }
		
	@abstractmethod
	def build_state(self, info):
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
		return np.zeros(self._shape, dtype=np.uint8)
		
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
		
class M_P_D_S_StateGenerator(StateGenerator): # 1 situation
	def _set_shape(self):
		self._shape = (4, 22, 80) # [heigth, width, channel]

	def build_state(self, info):
		state = self.empty_state()
		# layer 0: the map
		state = self.set_positions(state[0], info.get_list_of_walkable_positions(), 1)
		# layer 1: the player position
		state = self.set_positions(state[1], info.get_list_of_positions_by_tile("@"), 1)
		# layer 2: the doors positions
		state = self.set_positions(state[2], info.get_list_of_positions_by_tile("+"), 1)
		# layer 3: the stairs positions
		state = self.set_positions(state[3], info.get_list_of_positions_by_tile("%"), 1)
		return { "value" : state, "situation" : 0 }
		
class SingleLayer_StateGenerator(StateGenerator): # 1 situation
	def _set_shape(self):
		self._shape = (22, 80, 1) # [heigth, width, channel]
		
	def build_state(self, info):
		state = self.empty_state()
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("@"), 2) # rogue (player)
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("%"), 4) # stairs
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("|"), 8) # walls
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("-"), 8) # walls
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("+"), 16) # doors
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("#"), 16) # tunnel
		return state, 0
		
class TripleSituation_StateGenerator(StateGenerator): # 3 situations
	def _set_shape(self):
		self._shape = (22, 80, 1) # [heigth, width, channel]
		
	def _set_situations_count(self):
		self._situations = 3
		
	def build_state(self, info):
		state = self.empty_state()
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("%"), 4) # stairs
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("|"), 8) # walls
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("-"), 8) # walls
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("+"), 16) # doors
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("#"), 16) # tunnel
		# set it for last otherwise it may be overwritten by other positions!
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("@"), 2) # rogue (player)
			
		pixel = info.get_tile_below_player()
		if pixel == '#': # tunnel
			situation = 0
		elif pixel == "%": # stairs
			situation = 1
		else:
			situation = 2
		return state, situation
		
class TripleSituation_1_StateGenerator(TripleSituation_StateGenerator): # 5 situations
	def _set_shape(self):
		self._shape = (22, 80, 2) # [heigth, width, channel]
		
	def _set_situations_count(self):
		self._situations = 5
		
	def build_state(self, info):
		state = self.empty_state()
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("%"), 4) # stairs
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("|"), 2) # walls
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("-"), 2) # walls
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("+"), 1) # doors
		state = self.set_channel(0, state, info.get_list_of_positions_by_tile("#"), 1) # tunnel
		# set it for last otherwise it may be overwritten by other positions!
		state = self.set_channel(1, state, info.get_list_of_positions_by_tile("@"), 1) # rogue (player)
			
		pixel = info.get_tile_below_player()
		if pixel == '#': # situation 1
			return state, 0
		if pixel == "%": # situation 3
			return state, 1
			
		if info.get_tile_count("%") > 0: # situation 4
			return state, 2

		if self.environment_tiles_are_in_position_range(info, "|-", info.get_player_pos( ), 1): # situation 5
			return state, 3
		return state, 4
				
class CroppedView_StateGenerator(StateGenerator): # 6 situations

	def compute_state(self, info):
		self.player_position = info.get_player_pos( )
		if info.has_statusbar() and self.player_position != None:
			value, situation = self.build_state(info)
			return { "value" : value, "situation" : situation }
		return { "value" : self.empty_state(), "situation" : 0 }
		
	def _get_relative_coordinates(self, tile_position, centre_position, range):
		i, j = tile_position
		x, y = centre_position
		norm_i = i-x+floor(range[0]/2)
		norm_j = j-y+floor(range[1]/2)
		return norm_i, norm_j
		
	def set_channel(self, channel, centre_position, state, positions, value):
		for pos in positions:
			if pos:
				i, j = self._get_relative_coordinates(pos, centre_position, self._shape)
				if i >= 0 and j >= 0 and i < self._shape[0] and j < self._shape[1]:
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
		
class CroppedView_1_StateGenerator(CroppedView_StateGenerator): # 5 situations

	def _set_situations_count(self):
		self._situations = 5
		
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
		if pixel == "%": # situation 3
			return state, 1
			
		if info.get_tile_count("%") > 0: # situation 4
			return state, 2

		if self.environment_tiles_are_in_position_range(info, "|-", self.player_position, 1): # situation 5
			return state, 3
		return state, 4

class CroppedView_1b_StateGenerator(CroppedView_StateGenerator): # 4 situations

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


class CroppedView_1b_2L_StateGenerator(CroppedView_StateGenerator): # 4 situations
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