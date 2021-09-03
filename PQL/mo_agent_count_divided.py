from PQL.utils import Reward
from mo_agent import MOGridWorldAgent
from numpy.typing import NDArray
import numpy as np

class MOGridWorldAgentCountDivided(MOGridWorldAgent):

    def step_env(self, obs: NDArray[int], episode: int) -> tuple[NDArray[int], Reward, bool, int]:
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
        action_values = self.hv.compute(self.qsets(obs))

        for a in range(len(action_values)):
            action_values[a] /= (self.nsas[obs[0], obs[1], a] + 1)
            if action_values[a] == 0.:
                action_values[a] = float("inf") # States which have not been fully explored are interesting
        biggest_hvs = np.argwhere(action_values == np.amax(action_values)).flatten()
        if len(biggest_hvs) == 1:
            return biggest_hvs[0]
        else:
            # If there are equalities, randomly chooses among the equal pareto fronts
            return biggest_hvs[self.rng.integers(low=0, high=len(biggest_hvs))]
