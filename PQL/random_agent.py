from mo_env.deep_sea_treasure import DeepSeaTreasure
from mo_agent import MOGridWorldAgent
from numpy.typing import NDArray
import numpy as np

class RandomAgent(MOGridWorldAgent):
    def __init__(self, env: DeepSeaTreasure, num_episodes: int, interactive=False):
        super().__init__(env, num_episodes, interactive=interactive, mode='random')

    def heuristic(self, obs: NDArray[int]) -> int:
        """
            Diversifies by dividing by the number of times we already chose an action in a state
        """
        return self.env.sample_action()