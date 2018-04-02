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
		if x >= 0 and y >= 0 and x < len(self.map) and y < len(self.map[x]):
			return self.map[x][y]
		return ' '

	def get_player_pos(self):
		if len(self.pixel["agents"]["@"])>0:
			return self.pixel["agents"]["@"][0]
		print("Error: no agent @ visible on screen")
		return (-1,-1)
		
	def has_statusbar(self):
		return not self.statusbar["is_empty"]
		
	def get_list_of_positions_by_tile( self, tile ):
		for key in self.pixel:
			if self.pixel[key].get(tile):
				return self.pixel[key][tile]
		return []
		
	def get_list_of_positions_by_type( self, type ):
		result = []
		type_list = self.pixel[type] 
		for key in type_list:
			result = list( set().union ( result, type_list[key] ) )
		return result
		
	def get_list_of_walkable_positions(self):
		passages = self.get_list_of_positions_by_tile("#")
		doors = self.get_list_of_positions_by_tile("+")
		floors = self.get_list_of_positions_by_tile(".")
		items = self.get_list_of_positions_by_type("items")
		return list(set().union(passages, doors, floors, items))

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