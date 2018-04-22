
from roguelib_module.rewards import RewardGenerator
from roguelib_module.rewards import StairSeeker_RewardGenerator
from roguelib_module.rewards import ImprovedStairSeeker_RewardGenerator


class ImprovedStairSeeker_desc17_RewardGenerator(RewardGenerator):
    """Generate a reward for the last action:
        +17 for descending the stairs
        +1 for discovering new doors
        +1 for stepping on a door and discovering corridor tiles
        -0.01 for standing still
    """

    def get_value(self, frame_history):
        old_info = frame_history[-2]
        new_info = frame_history[-1]
        if new_info.statusbar["dungeon_level"] > old_info.statusbar["dungeon_level"]:
            self.goal_achieved = True
            return 17
        elif new_info.get_tile_count("+") > old_info.get_tile_count("+"):  # doors
            return 1
        if old_info.get_tile_below_player() == '+':
            if new_info.get_tile_count("#") > old_info.get_tile_count("#"):  # has started to explore
                return 1
        elif self.player_standing_still(old_info, new_info):  # standing reward
            return -0.01
        return 0


class ImprovedStairSeeker_2_RewardGenerator(StairSeeker_RewardGenerator):

    def get_value(self, frame_history):
        old_info = frame_history[-2]
        new_info = frame_history[-1]
        if old_info.get_tile_below_player() == '+' and new_info.get_tile_count("#") > old_info.get_tile_count("#"):  # has opened a new door
            return 0.5
        return super().get_value(frame_history)


class ImprovedStairSeeker_3_RewardGenerator(StairSeeker_RewardGenerator):

    def get_value(self, frame_history):
        old_info = frame_history[-2]
        new_info = frame_history[-1]
        if old_info.get_tile_below_player() == '+' and new_info.get_tile_count("#") > old_info.get_tile_count("#"):  # has opened a new door
            return 0.25
        return super().get_value(frame_history)


class BalancedStairSeeker_RewardGenerator(RewardGenerator):

    def get_value(self, frame_history):
        old_info = frame_history[-2]
        new_info = frame_history[-1]
        if new_info.statusbar["dungeon_level"] > old_info.statusbar["dungeon_level"]:
            self.goal_achieved = True
            return 10
        if old_info.get_tile_below_player() == '+' and new_info.get_tile_count("#") > old_info.get_tile_count("#"):  # has opened a new door
            return 0.125
        elif new_info.get_tile_count("+") > old_info.get_tile_count("+"):  # doors
            return 0.5
        elif self.player_standing_still(old_info, new_info):  # standing reward
            return -0.01
        return 0


class SuperStairSeeker_RewardGenerator(RewardGenerator):

    def get_value(self, frame_history):
        old_info = frame_history[-2]
        new_info = frame_history[-1]
        if new_info.statusbar["dungeon_level"] > old_info.statusbar["dungeon_level"]:
            self.goal_achieved = True
            return 100
        elif old_info.get_tile_below_player() == '+' and new_info.get_tile_count("#") > old_info.get_tile_count("#"):  # has opened a new door
            return 1
        elif new_info.get_tile_count("+") > old_info.get_tile_count("+"):  # doors
            return 1
        elif self.player_standing_still(old_info, new_info):  # standing reward
            return -0.01
        return 0