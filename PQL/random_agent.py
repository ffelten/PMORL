from mo_agent import MOGridWorldAgent
from numpy.typing import NDArray
import numpy as np

class RandomAgent(MOGridWorldAgent):
    def heuristic(self, obs: NDArray[int]) -> int:
        """
            Diversifies by dividing by the number of times we already chose an action in a state
        """
        return self.env.sample_action()