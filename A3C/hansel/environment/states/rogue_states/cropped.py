
from rogueinabox.states import CroppedView_2b_2L_StateGenerator


class GetShapeMixin:
    def get_shape(self):
        return self._shape


class CroppedView_2L_17x17_StateGenerator(GetShapeMixin, CroppedView_2b_2L_StateGenerator):
    """Generates a 17x17 state composed of a single layer cropped and centered around the rouge containing:
        - stairs
        - walls
        - doors and corridors
        The numerical values used are different and the rogue is not directly shown in the state.
    """
    pass
