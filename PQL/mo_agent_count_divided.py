import utils.argmax
import utils.softmax
from mo_env.deep_sea_treasure import DeepSeaTreasure
from utils import Reward
from mo_agent import MOGridWorldAgent
from numpy.typing import NDArray
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

class MOGridWorldAgentCountDivided(MOGridWorldAgent):
    """
    1/count(state,action), no exploitation
    """

    def __init__(self, env: DeepSeaTreasure, num_episodes: int, interactive=False, count_weight: int = 1., he_weight = 1.):
        super().__init__(env, num_episodes, mode='count_divided', interactive=interactive)
        self.count_weight = count_weight
        self.he_weight = he_weight

    def step_env(self, obs: NDArray[int], timestep: int) -> tuple[NDArray[int], Reward, bool, int]:
        """
        Overrides step_env to remove the e-greedy meta heuristic
        """
        best_action = self.heuristic(obs)
        chosen_action = best_action

        obs, reward, done = self.env.step(chosen_action)
        return obs, reward, done, chosen_action

    def heuristic(self, obs: NDArray[int]) -> int:
        """
            Diversifies by dividing by the number of times we already chose an action in a state
        """
        action_values = np.ones_like(self.qsets(obs))


        for a in range(len(action_values)):
            # reduce the score by the number of times it has already been done
            action_values[a] /= (self.nsas[obs[0], obs[1], a] + 1) ** self.count_weight
            if self.nsas[obs[0], obs[1], a] == 0:
                action_values[a] = 2**256 # optimistic first lookup, curiosity


        return utils.argmax.argmax(action_values)
