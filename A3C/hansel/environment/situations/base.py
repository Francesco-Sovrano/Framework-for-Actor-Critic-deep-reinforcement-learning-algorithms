from abc import ABC, abstractmethod


class SituationGenerator(ABC):
    """Base class for situation generators"""

    def __init__(self):
        self.reset()

    def reset(self):
        pass

    @abstractmethod
    def situations_count(self):
        """Returns the number of situations of the generator"""
        pass

    @abstractmethod
    def compute_situation(self, frame_history):
        """Returns a unique identifier of the current situation

        :param frame_history:
             list of frames the situation should be computed on
        :rtype: int
        :return:
            numerical identifier of the situation
        """
        pass


class Dummy_SituationGenerator(SituationGenerator):
    """Dummy generator that always returns the same situation"""

    def situations_count(self):
        return 1

    def compute_situation(self, frame_history):
        return 0
