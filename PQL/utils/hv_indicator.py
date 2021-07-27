import numpy as np
from numpy.typing import NDArray

from pygmo import hypervolume

class MaxHVHeuristic:
    """
    Hypervolume based heuristic
    """

    def __init__(
            self,
            ref_point: NDArray[np.float64],
            seed=42
    ):
        self.ref_point = ref_point
        self.rng = np.random.default_rng(seed)

    def compute(self, action_sets) -> NDArray[float]:
        """
        Computes the biggest hypervolume to choose
        :param: qsets: array of qsets for each possible action
        :return: the hypervolume values for each action
        """
        action_values = np.zeros(action_sets.shape[0])
        for action in range(len(action_values)):
            qset = action_sets[action].set
            if not qset:
                hv = 0.
            else:
                hv = hypervolume(qset).compute(self.ref_point)
            action_values[action] = hv

        return action_values

