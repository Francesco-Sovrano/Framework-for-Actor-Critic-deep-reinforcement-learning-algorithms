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
		self.reset()
		
	def reset(self):
		self.need_reset = False

	@abstractmethod
	def _set_shape(self):
		"""The implementing class MUST set the state _shape (should be a tuple)."""
		self._shape = (0, 0, 0)

	@abstractmethod
	def compute_state(self, info):
		"""Should compute the state and return it."""
		pass

	def set_layer(self, state, layer, positions, value):
		for pos in positions:
			if pos:
				i, j = pos
				state[layer][i][j] = value
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
		
class SingleLayer_StateGenerator(StateGenerator):
	def _set_shape(self):
		self._shape = (1, 22, 80)
		
	def compute_state(self, info):
		state = self.empty_state()
		if info.has_statusbar():
			state = self.set_layer(state, 0, info.get_list_of_positions_by_tile("@"), 2) # rogue (player)
			state = self.set_layer(state, 0, info.get_list_of_positions_by_tile("%"), 4) # stairs
			state = self.set_layer(state, 0, info.get_list_of_positions_by_tile("|"), 8) # walls
			state = self.set_layer(state, 0, info.get_list_of_positions_by_tile("-"), 8) # walls
			state = self.set_layer(state, 0, info.get_list_of_positions_by_tile("+"), 16) # doors
			state = self.set_layer(state, 0, info.get_list_of_positions_by_tile("#"), 16) # tunnel
		return { "environment" : state, "layer" : 0 }
		
class TripleLayer_StateGenerator(StateGenerator):
	def _set_shape(self):
		self._shape = (3, 22, 80)
		
	def compute_state(self, info):
		environment = self.empty_state()
		layer = 2
		if info.has_statusbar():
			#layer 1
			environment = self.set_layer(environment, 0, info.get_list_of_positions_by_tile("#"), 1) # tunnel
			#layer 2
			environment = self.set_layer(environment, 1, info.get_list_of_positions_by_tile("%"), 1) # stairs
			#layer 3
			environment = self.set_layer(environment, 2, info.get_list_of_positions_by_tile("."), 1) # floor
			environment = self.set_layer(environment, 2, info.get_list_of_positions_by_tile("+"), 1) # doors
			
			pixel = info.get_tile_below_player()
			if pixel == '#': # tunnel
				environment = self.set_layer(environment, 0, info.get_list_of_positions_by_tile("@"), 2)
				layer = 0
			elif pixel == "%": # stairs
				environment = self.set_layer(environment, 1, info.get_list_of_positions_by_tile("@"), 2)
				layer = 1
			else: # floor
				environment = self.set_layer(environment, 2, info.get_list_of_positions_by_tile("@"), 2)
		return { "environment" : environment, "layer" : layer }
		
class TripleLayer_1_StateGenerator(TripleLayer_StateGenerator):
	def compute_state(self, info):
		state = self.empty_state()
		layer = 2
		if info.has_statusbar():
			
			#layer 1
			state = self.set_layer(state, 0, info.get_list_of_positions_by_tile("%"), 4) # stairs
			state = self.set_layer(state, 0, info.get_list_of_positions_by_tile("|"), 8) # walls
			state = self.set_layer(state, 0, info.get_list_of_positions_by_tile("-"), 8) # walls
			state = self.set_layer(state, 0, info.get_list_of_positions_by_tile("+"), 16) # doors
			state = self.set_layer(state, 0, info.get_list_of_positions_by_tile("#"), 16) # tunnel
			#layer 2
			state = self.set_layer(state, 1, info.get_list_of_positions_by_tile("%"), 4) # stairs
			state = self.set_layer(state, 1, info.get_list_of_positions_by_tile("|"), 8) # walls
			state = self.set_layer(state, 1, info.get_list_of_positions_by_tile("-"), 8) # walls
			state = self.set_layer(state, 1, info.get_list_of_positions_by_tile("+"), 16) # doors
			state = self.set_layer(state, 1, info.get_list_of_positions_by_tile("#"), 16) # tunnel
			#layer 3
			state = self.set_layer(state, 2, info.get_list_of_positions_by_tile("%"), 4) # stairs
			state = self.set_layer(state, 2, info.get_list_of_positions_by_tile("|"), 8) # walls
			state = self.set_layer(state, 2, info.get_list_of_positions_by_tile("-"), 8) # walls
			state = self.set_layer(state, 2, info.get_list_of_positions_by_tile("+"), 16) # doors
			state = self.set_layer(state, 2, info.get_list_of_positions_by_tile("#"), 16) # tunnel
			
			pixel = info.get_tile_below_player()
			if pixel == '#': # tunnel
				state = self.set_layer(state, 0, info.get_list_of_positions_by_tile("@"), 2) # rogue (player), set it for last otherwise it may be overwritten by other positions!
				layer = 0
			elif pixel == "%": # stairs
				state = self.set_layer(state, 1, info.get_list_of_positions_by_tile("@"), 2) # rogue (player), set it for last otherwise it may be overwritten by other positions!
				layer = 1
			else: # floor
				state = self.set_layer(state, 2, info.get_list_of_positions_by_tile("@"), 2) # rogue (player), set it for last otherwise it may be overwritten by other positions!
		return { "environment" : state, "layer" : layer }
		
class TripleLayer_2_StateGenerator(TripleLayer_StateGenerator):
	def compute_state(self, info):
		environment = self.empty_state()
		layer = 2
		if info.has_statusbar():
			
			#layer 1
			environment = self.set_layer(environment, 0, info.get_list_of_positions_by_tile("#"), 1) # tunnel
			environment = self.set_layer(environment, 0, info.get_list_of_positions_by_tile("+"), 1) # doors
			#layer 2
			environment = self.set_layer(environment, 1, info.get_list_of_positions_by_tile("%"), 1) # stairs
			#layer 3
			environment = self.set_layer(environment, 2, info.get_list_of_positions_by_tile("-"), 1) # walls
			environment = self.set_layer(environment, 2, info.get_list_of_positions_by_tile("|"), 1) # walls
			environment = self.set_layer(environment, 2, info.get_list_of_positions_by_tile("%"), 2) # stairs
			
			pixel = info.get_tile_below_player()
			if pixel == '#': # tunnel
				environment = self.set_layer(environment, 0, info.get_list_of_positions_by_tile("@"), 8)
				layer = 0
			elif pixel == "%": # stairs
				environment = self.set_layer(environment, 1, info.get_list_of_positions_by_tile("@"), 8)
				layer = 1
			else: # floor
				environment = self.set_layer(environment, 2, info.get_list_of_positions_by_tile("@"), 8)
		return { "environment" : environment, "layer" : layer }
				
class CroppedView_StateGenerator(StateGenerator):
	def __init__(self):
		super().__init__()
		for i in range(1, len(self._shape)): 
			if self._shape[i] % 2 == 0: # there should be always a center, thus each layer dimension should be even
				self._shape[i] += 1
		
	def _get_relative_coordinates(self, tile_position, centre_position, range):
		i, j = tile_position
		x, y = centre_position
		norm_i = i-x+floor(range[1]/2)
		norm_j = j-y+floor(range[2]/2)
		return norm_i, norm_j
		
	def set_layer(self, centre_position, state, layer, positions, value):
		for pos in positions:
			if pos:
				i, j = self._get_relative_coordinates(pos, centre_position, self._shape)
				if i >= 0 and j >= 0 and i < self._shape[1] and j < self._shape[2]:
					state[layer][i][j] = value
		return state
		
	def _set_shape(self):
		self._shape = (3, 11, 11)
		
	def compute_state(self, info):
		environment = self.empty_state()
		layer = 2
		player_position = info.get_player_pos( )
		if info.has_statusbar() and player_position != None:
			
			#layer 1
			environment = self.set_layer(player_position, environment, 0, info.get_list_of_positions_by_tile("#"), 1) # tunnel
			environment = self.set_layer(player_position, environment, 0, info.get_list_of_positions_by_tile("+"), 1) # doors
			#layer 2
			environment = self.set_layer(player_position, environment, 1, info.get_list_of_positions_by_tile("%"), 1) # stairs
			#layer 3
			environment = self.set_layer(player_position, environment, 2, info.get_list_of_positions_by_tile("-"), 1) # walls
			environment = self.set_layer(player_position, environment, 2, info.get_list_of_positions_by_tile("|"), 1) # walls
			environment = self.set_layer(player_position, environment, 2, info.get_list_of_positions_by_tile("%"), 2) # stairs
			
			pixel = info.get_tile_below_player()
			if pixel == '#': # tunnel
				layer = 0
			elif pixel == "%": # stairs
				layer = 1
				
		# file = open( '/public/francesco_sovrano/states_debug_info.log',"w") 
		# for x in range(self._shape[0]):
			# for y in range(self._shape[1]):
				# for z in range(self._shape[2]):
					# file.write( str(environment[x][y][z]) )
				# file.write( '\n' )
			# file.write( '\n' )
		# file.close()
		return { "environment" : environment, "layer" : layer }

class CroppedView_1_StateGenerator(CroppedView_StateGenerator):
	def _set_shape(self):
		self._shape = (3, 17, 17)
		
class CroppedView_4_StateGenerator(CroppedView_StateGenerator):
	def _set_shape(self):
		self._shape = (6, 17, 17)
		
	def compute_state(self, info):
		state = self.empty_state()
		player_position = info.get_player_pos( )
		if info.has_statusbar() and player_position != None:
			#layer 1
			state = self.set_layer(player_position, state, 0, info.get_list_of_positions_by_tile("%"), 4) # stairs
			state = self.set_layer(player_position, state, 0, info.get_list_of_positions_by_tile("|"), 8) # walls
			state = self.set_layer(player_position, state, 0, info.get_list_of_positions_by_tile("-"), 8) # walls
			state = self.set_layer(player_position, state, 0, info.get_list_of_positions_by_tile("+"), 16) # doors
			state = self.set_layer(player_position, state, 0, info.get_list_of_positions_by_tile("#"), 16) # tunnel
			#layer 2
			state = self.set_layer(player_position, state, 1, info.get_list_of_positions_by_tile("%"), 4) # stairs
			state = self.set_layer(player_position, state, 1, info.get_list_of_positions_by_tile("|"), 8) # walls
			state = self.set_layer(player_position, state, 1, info.get_list_of_positions_by_tile("-"), 8) # walls
			state = self.set_layer(player_position, state, 1, info.get_list_of_positions_by_tile("+"), 16) # doors
			state = self.set_layer(player_position, state, 1, info.get_list_of_positions_by_tile("#"), 16) # tunnel
			#layer 3
			state = self.set_layer(player_position, state, 2, info.get_list_of_positions_by_tile("%"), 4) # stairs
			state = self.set_layer(player_position, state, 2, info.get_list_of_positions_by_tile("|"), 8) # walls
			state = self.set_layer(player_position, state, 2, info.get_list_of_positions_by_tile("-"), 8) # walls
			state = self.set_layer(player_position, state, 2, info.get_list_of_positions_by_tile("+"), 16) # doors
			state = self.set_layer(player_position, state, 2, info.get_list_of_positions_by_tile("#"), 16) # tunnel
			#layer 4
			state = self.set_layer(player_position, state, 3, info.get_list_of_positions_by_tile("%"), 4) # stairs
			state = self.set_layer(player_position, state, 3, info.get_list_of_positions_by_tile("|"), 8) # walls
			state = self.set_layer(player_position, state, 3, info.get_list_of_positions_by_tile("-"), 8) # walls
			state = self.set_layer(player_position, state, 3, info.get_list_of_positions_by_tile("+"), 16) # doors
			state = self.set_layer(player_position, state, 3, info.get_list_of_positions_by_tile("#"), 16) # tunnel
			#layer 5
			state = self.set_layer(player_position, state, 4, info.get_list_of_positions_by_tile("%"), 4) # stairs
			state = self.set_layer(player_position, state, 4, info.get_list_of_positions_by_tile("|"), 8) # walls
			state = self.set_layer(player_position, state, 4, info.get_list_of_positions_by_tile("-"), 8) # walls
			state = self.set_layer(player_position, state, 4, info.get_list_of_positions_by_tile("+"), 16) # doors
			state = self.set_layer(player_position, state, 4, info.get_list_of_positions_by_tile("#"), 16) # tunnel
			#layer 6
			state = self.set_layer(player_position, state, 5, info.get_list_of_positions_by_tile("%"), 4) # stairs
			state = self.set_layer(player_position, state, 5, info.get_list_of_positions_by_tile("|"), 8) # walls
			state = self.set_layer(player_position, state, 5, info.get_list_of_positions_by_tile("-"), 8) # walls
			state = self.set_layer(player_position, state, 5, info.get_list_of_positions_by_tile("+"), 16) # doors
			state = self.set_layer(player_position, state, 5, info.get_list_of_positions_by_tile("#"), 16) # tunnel
			
			pixel = info.get_tile_below_player()
			if pixel == '#': # layer 1
				return { "environment" : state, "layer" : 0 }
			if pixel == '+': # layer 2
				return { "environment" : state, "layer" : 1 }
			if pixel == "%": # layer 3
				return { "environment" : state, "layer" : 2 }
				
			if info.get_tile_count("%") > 0: # layer 4
				return { "environment" : state, "layer" : 3 }

			if self.environment_tiles_are_in_position_range(info, "|-", player_position, 1): # layer 5
				return { "environment" : state, "layer" : 4 }
		return { "environment" : state, "layer" : 5 }  # layer 6
		
class CroppedView_5_StateGenerator(CroppedView_StateGenerator):
	def _set_shape(self):
		self._shape = (5, 17, 17)
		
	def compute_state(self, info):
		state = self.empty_state()
		player_position = info.get_player_pos()
		if info.has_statusbar() and player_position != None:
			#layer 1
			state = self.set_layer(player_position, state, 0, info.get_list_of_positions_by_tile("+"), 1) # doors
			state = self.set_layer(player_position, state, 0, info.get_list_of_positions_by_tile("#"), 1) # passages
			state = self.set_layer(player_position, state, 0, info.get_list_of_positions_by_tile("|"), 2) # walls
			state = self.set_layer(player_position, state, 0, info.get_list_of_positions_by_tile("-"), 2) # walls
			#layer 2
			state = self.set_layer(player_position, state, 1, info.get_list_of_positions_by_tile("%"), 1) # stairs
			#layer 3
			state = self.set_layer(player_position, state, 2, info.get_list_of_positions_by_tile("%"), 1) # stairs
			state = self.set_layer(player_position, state, 2, info.get_list_of_positions_by_tile("|"), 2) # walls
			state = self.set_layer(player_position, state, 2, info.get_list_of_positions_by_tile("-"), 2) # walls
			#layer 4
			state = self.set_layer(player_position, state, 3, info.get_list_of_positions_by_tile("|"), 1) # walls
			state = self.set_layer(player_position, state, 3, info.get_list_of_positions_by_tile("-"), 1) # walls
			#layer 5
			state = self.set_layer(player_position, state, 4, info.get_list_of_positions_by_tile("|"), 1) # walls
			state = self.set_layer(player_position, state, 4, info.get_list_of_positions_by_tile("-"), 1) # walls
			
			pixel = info.get_tile_below_player()
			if pixel in '#+': # layer 1
				return { "environment" : state, "layer" : 0 }
			if pixel == "%": # layer 2
				return { "environment" : state, "layer" : 1 }
			if info.get_tile_count("%") > 0: # layer 3
				return { "environment" : state, "layer" : 2 }
			if self.environment_tiles_are_in_position_range(info, "|-", player_position, 1): # layer 4
				return { "environment" : state, "layer" : 3 }
				
		return { "environment" : state, "layer" : 4 } # layer 5