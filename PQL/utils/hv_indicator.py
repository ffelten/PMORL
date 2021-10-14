import numpy as np
from numpy.typing import NDArray

from pygmo import hypervolume

class MaxHVHeuristic:
    """
    Hypervolume based heuristic
    """

    def __init__(
            self,
            ref_point: NDArray[np.float64]
    ):
        self.ref_point = ref_point

    def compute(self, action_sets) -> NDArray[float]:
        """
        Computes the biggest hypervolume to choose
        :param: qsets: array of qsets for each possible action
        :return: the hypervolume values for each action
        """
        action_values = np.zeros(action_sets.shape[0])
        for action in range(len(action_values)):
            qset = action_sets[action].set
            if len(qset) == 1 and np.all(qset[0] == np.zeros_like(qset[0])):
                hv = 0.
            else:
                ## Negates because we maximize and the hv computation supposes minimization
                negated_qset = np.array(qset) * -1.
                hv = hypervolume(negated_qset).compute(self.ref_point)
            action_values[action] = hv

        return action_values

