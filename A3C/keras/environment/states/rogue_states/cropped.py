
from roguelib_module.states import CroppedView_Base_StateGenerator
from roguelib_module.states import CroppedView_SingleLayer_17x17_StateGenerator


class CroppedView_2L_17x17_StateGenerator(CroppedView_SingleLayer_17x17_StateGenerator):
    """Generates a 17x17 state composed of a single layer cropped and centered around the rouge containing:
        - stairs
        - walls
        - doors and corridors
        The numerical values used are different and the rogue is not directly shown in the state.
    """

    def _set_shape(self, data_format):
        self._shape = (2, 17, 17) if data_format == "channels_first" else (17, 17, 2)

    def build_state(self, current_frame, frame_history):
        state = self.empty_state()
        player_position = current_frame.get_list_of_positions_by_tile("@")[0]

        self.set_channel_relative(player_position, state, 1, current_frame.get_list_of_positions_by_tile("%"), 4)  # stairs
        self.set_channel_relative(player_position, state, 0, current_frame.get_list_of_positions_by_tile("|"), 8)  # walls
        self.set_channel_relative(player_position, state, 0, current_frame.get_list_of_positions_by_tile("-"), 8)  # walls
        self.set_channel_relative(player_position, state, 0, current_frame.get_list_of_positions_by_tile("+"), 16)  # doors
        self.set_channel_relative(player_position, state, 0, current_frame.get_list_of_positions_by_tile("#"), 16)  # tunnel

        return state
