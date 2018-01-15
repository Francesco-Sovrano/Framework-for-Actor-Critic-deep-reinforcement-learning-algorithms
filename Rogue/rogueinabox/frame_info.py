class RogueFrameInfo:
	def __init__(self, pixel, map, statusbar, screen):
		self.pixel = pixel
		self.map = map
		self.statusbar = statusbar
		self.screen = screen
		
	def get_tile_below_player(self):
		pos = self.get_player_pos( )
		return self.get_environment_tile_at(pos)
		
	def get_environment_tile_at(self, pos):
		x, y = pos
		return self.map[x][y]

	def get_player_pos(self):
		return self.pixel["agents"]["@"][0]
		
	def has_statusbar(self):
		return not self.statusbar["is_empty"]
		
	def get_list_of_positions_by_tile( self, tile ):
		for key in self.pixel:
			if self.pixel[key].get(tile):
				return self.pixel[key][tile]
		return []
		
	def get_tile_count( self, tile ):
		return len( self.get_list_of_positions_by_tile(tile) )
				
	def get_type_count( self, type ):
		count = 0
		dict = self.pixel[type]
		for tile in dict:
			count += len(dict[tile])
		return count
				
	def get_known_tiles_count( self ):
		count = 0
		for key in self.pixel:
			count += self.get_type_count(key)
		return count