
from roguelib_module.states import StateGenerator, CroppedViewOnRogue_Base_StateGenerator


class StateSituationGenerator(StateGenerator):
    """State generator that also returns a situation

    N.B. self.build_state() must return a pair (value, situation)
    """
    def _subinit(self):
        super()._subinit()
        self._set_situations_count()

    def _set_situations_count(self):
        self._situations = 0

    def compute_state(self, frame_history):
        if self.is_frame_history_sufficient(frame_history):
            value, situation = self.build_state(frame_history[-1], frame_history)
        else:
            value, situation = self.empty_state(), 0
        return {"value": value, "situation": situation}

    @staticmethod
    def environment_tiles_are_in_position_range(info, tiles, position, r):
        x, y = position
        for i in range(-r, r + 1):
            a = x + i
            for j in range(-r, r + 1):
                b = y + j
                pixel = info.get_environment_tile_at((a, b))
                if pixel in tiles:
                    return True
        return False


class CroppedView_StateSituationGenerator(StateSituationGenerator, CroppedViewOnRogue_Base_StateGenerator):
    pass


class CroppedView_1_StateGenerator(CroppedView_StateSituationGenerator):  # 5 situations

    def _set_shape(self, data_format):
        self._shape = (1, 17, 17) if data_format == "channels_first" else (17, 17, 1)

    def _set_situations_count(self):
        self._situations = 5

    def build_state(self, info, history):
        state = self.empty_state()
        state = self.set_channel_relative(self.player_position, state, 0, info.get_list_of_positions_by_tile("%"), 4)  # stairs
        state = self.set_channel_relative(self.player_position, state, 0, info.get_list_of_positions_by_tile("|"), 8)  # walls
        state = self.set_channel_relative(self.player_position, state, 0, info.get_list_of_positions_by_tile("-"), 8)  # walls
        state = self.set_channel_relative(self.player_position, state, 0, info.get_list_of_positions_by_tile("+"), 16)  # doors
        state = self.set_channel_relative(self.player_position, state, 0, info.get_list_of_positions_by_tile("#"), 16)  # tunnel

        pixel = info.get_tile_below_player()
        if pixel == '#':  # situation 1
            return state, 0
        if pixel == "%":  # situation 3
            return state, 1

        if info.get_tile_count("%") > 0:  # situation 4
            return state, 2

        if self.environment_tiles_are_in_position_range(info, "|-", self.player_position, 1):  # situation 5
            return state, 3
        return state, 4


class CroppedView_1b_StateGenerator(CroppedView_StateSituationGenerator):  # 4 situations

    def _set_shape(self, data_format):
        self._shape = (1, 17, 17) if data_format == "channels_first" else (17, 17, 1)

    def _set_situations_count(self):
        self._situations = 4

    def build_state(self, info, history):
        state = self.empty_state()
        state = self.set_channel_relative(self.player_position, state, 0, info.get_list_of_positions_by_tile("%"), 4)  # stairs
        state = self.set_channel_relative(self.player_position, state, 0, info.get_list_of_positions_by_tile("|"), 8)  # walls
        state = self.set_channel_relative(self.player_position, state, 0, info.get_list_of_positions_by_tile("-"), 8)  # walls
        state = self.set_channel_relative(self.player_position, state, 0, info.get_list_of_positions_by_tile("+"), 16)  # doors
        state = self.set_channel_relative(self.player_position, state, 0, info.get_list_of_positions_by_tile("#"), 16)  # tunnel

        pixel = info.get_tile_below_player()
        if pixel == '#':  # situation 0
            return state, 0

        if info.get_tile_count("%") > 0:  # situation 1
            return state, 1

        if self.environment_tiles_are_in_position_range(info, "|-", self.player_position, 1):  # situation 2
            return state, 2
        return state, 3


class CroppedView_1b_2L_StateGenerator(CroppedView_StateSituationGenerator):  # 4 situations

    def _set_shape(self, data_format):
        self._shape = (2, 17, 17) if data_format == "channels_first" else (17, 17, 2)

    def _set_situations_count(self):
        self._situations = 4

    def build_state(self, info, history):
        state = self.empty_state()
        state = self.set_channel_relative(self.player_position, state, 1, info.get_list_of_positions_by_tile("%"), 4)  # stairs
        state = self.set_channel_relative(self.player_position, state, 0, info.get_list_of_positions_by_tile("|"), 8)  # walls
        state = self.set_channel_relative(self.player_position, state, 0, info.get_list_of_positions_by_tile("-"), 8)  # walls
        state = self.set_channel_relative(self.player_position, state, 0, info.get_list_of_positions_by_tile("+"), 16)  # doors
        state = self.set_channel_relative(self.player_position, state, 0, info.get_list_of_positions_by_tile("#"), 16)  # tunnel

        pixel = info.get_tile_below_player()
        if pixel == '#':  # situation 0
            return state, 0

        if info.get_tile_count("%") > 0:  # situation 1
            return state, 1

        if self.environment_tiles_are_in_position_range(info, "|-", self.player_position, 1):  # situation 2
            return state, 2
        return state, 3


class CroppedView_2b_2L_StateGenerator(CroppedView_StateSituationGenerator):  # 4 situations

    def _set_shape(self, data_format):
        self._shape = (2, 17, 17) if data_format == "channels_first" else (17, 17, 2)

    def _set_situations_count(self):
        self._situations = 2

    def build_state(self, info, history):
        state = self.empty_state()
        state = self.set_channel_relative(self.player_position, state, 1, info.get_list_of_positions_by_tile("%"), 1)  # stairs
        state = self.set_channel_relative(self.player_position, state, 0, info.get_list_of_positions_by_tile("|"), 1)  # walls
        state = self.set_channel_relative(self.player_position, state, 0, info.get_list_of_positions_by_tile("-"), 1)  # walls
        state = self.set_channel_relative(self.player_position, state, 0, info.get_list_of_positions_by_tile("+"), 2)  # doors
        state = self.set_channel_relative(self.player_position, state, 0, info.get_list_of_positions_by_tile("#"), 2)  # tunnel

        pixel = info.get_tile_below_player()
        if info.get_tile_count("%") > 0:  # situation 1
            return state, 1
        return state, 0


class CroppedView_1b_3L_StateGenerator(CroppedView_StateSituationGenerator):  # 4 situations

    def _set_shape(self, data_format):
        self._shape = (3, 17, 17) if data_format == "channels_first" else (17, 17, 3)

    def _set_situations_count(self):
        self._situations = 4

    def build_state(self, info, history):
        state = self.empty_state()
        state = self.set_channel_relative(self.player_position, state, 1, info.get_list_of_positions_by_tile("%"), 4)  # stairs
        state = self.set_channel_relative(self.player_position, state, 2, info.get_list_of_positions_by_tile("|"), 8)  # walls
        state = self.set_channel_relative(self.player_position, state, 2, info.get_list_of_positions_by_tile("-"), 8)  # walls
        state = self.set_channel_relative(self.player_position, state, 0, info.get_list_of_positions_by_tile("+"), 16)  # doors
        state = self.set_channel_relative(self.player_position, state, 0, info.get_list_of_positions_by_tile("#"), 16)  # tunnel

        pixel = info.get_tile_below_player()
        if pixel == '#':  # situation 0
            return state, 0

        if info.get_tile_count("%") > 0:  # situation 1
            return state, 1

        if self.environment_tiles_are_in_position_range(info, "|-", self.player_position, 1):  # situation 2
            return state, 2
        return state, 3
