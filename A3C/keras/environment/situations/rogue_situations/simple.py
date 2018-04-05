from .base import Rogue_SituationGenerator, Dummy_SituationGenerator


class Single_SituationGenerator(Dummy_SituationGenerator):
    """Generates only 1 situation"""
    pass


class Stairs_SituationGenerator(Rogue_SituationGenerator):
    """Generates 2 situations, based on the visibility of the stairs:
        - stairs are visible
        - stairs are not visible
    """

    def situations_count(self):
        return 2

    def compute_situation(self, frame_history):
        current_frame = frame_history[-1]

        if current_frame.get_tile_count("%") > 0:
            # stairs are visible
            return 0
        # stairs are not visible
        return 1


class Corridors_Stairs_Walls_SituationGenerator(Rogue_SituationGenerator):
    """Generates 4 situations, based on the rogue position:
        - in a corridor
        - visible stairs
        - next to a wall
        - otherwise
    """

    def situations_count(self):
        return 4

    def compute_situation(self, frame_history):
        current_frame = frame_history[-1]

        rogue_position = current_frame.get_player_pos()
        below_rogue = current_frame.get_tile_below_player()

        if below_rogue == '#':
            # in a corridor
            return 0
        elif current_frame.get_tile_count("%") > 0:
            # stairs are visible
            return 1
        elif self.environment_tiles_are_in_position_range(current_frame, "|-", rogue_position, 1):
            # next to a wall
            return 2
        # any other situation
        return 3


class Corridors_OnStairs_Walls_SituationGenerator(Rogue_SituationGenerator):
    """Generates 5 situations, based on the rogue position:
        - in a corridor
        - on the stairs
        - visible stairs
        - next to a wall
        - otherwise
    """

    def situations_count(self):
        return 5

    def compute_situation(self, frame_history):
        current_frame = frame_history[-1]

        rogue_position = current_frame.get_player_pos()
        below_rogue = current_frame.get_tile_below_player()

        if below_rogue == '#':
            # in a corridor
            return 0
        elif below_rogue == "%":
            # on the stairs
            return 1
        elif current_frame.get_tile_count("%") > 0:
            # stairs are visible
            return 2
        elif self.environment_tiles_are_in_position_range(current_frame, "|-", rogue_position, 1):
            # next to a wall
            return 3
        # any other situation
        return 4
