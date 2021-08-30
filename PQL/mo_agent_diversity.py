from mo_agent import MOGridWorldAgent
from numpy.typing import NDArray
import numpy as np

class MOGridWorldAgentDiversity(MOGridWorldAgent):
    def heuristic(self, obs: NDArray[int]) -> int:
        """
            Diversifies by dividing by the number of times we already chose an action in a state
        """
        action_values = self.hv.compute(self.qsets[obs[0], obs[1]])
        for a in range(len(action_values)):
            action_values[a] /= (self.nsas[obs[0], obs[1], a] + 1)
        biggest_hvs = np.argwhere(action_values == np.amax(action_values)).flatten()
        if len(biggest_hvs) == 1:
            return biggest_hvs[0]
        else:
            # If there are equalities, randomly chooses among the equal pareto fronts
            return biggest_hvs[self.rng.integers(low=0, high=len(biggest_hvs))]
