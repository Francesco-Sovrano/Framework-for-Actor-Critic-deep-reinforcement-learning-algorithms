from abc import abstractmethod
from .. import SituationGenerator, Dummy_SituationGenerator


class Rogue_SituationGenerator(SituationGenerator):
    """Base class for rogue situation generators"""

    @abstractmethod
    def situations_count(self):
        """Returns the number of situations of the generator"""
        pass

    @abstractmethod
    def compute_situation(self, frame_history):
        """Returns a unique identifier of the current situation

        :param list[roguelib_module.frame_info.RogueFrameInfo] frame_history:
            list of frames the situation should be computed on
        :rtype: int
        :return:
            numerical identifier of the situation
        """
        pass

    @staticmethod
    def environment_tiles_are_in_position_range(info, tiles, position, r):
        """Returns whether any of the given tiles are in range 'r' of 'position'

        :param roguelib_module.frame_info.RogueFrameInfo info:
            frame to consider
        :param iterable tiles:
            list-like object containing the tiles looked for
        :param tuple[int,int] position:
            position around which to look for the tiles
        :param int r:
            range from position
        :rtype:
        :return:
            whether any of the given tiles are in range 'r' of 'position'
        """
        x, y = position
        for i in range(-r, r + 1):
            a = x + i
            for j in range(-r, r + 1):
                b = y + j
                pixel = info.get_environment_tile_at((a, b))
                if pixel in tiles:
                    return True
        return False
