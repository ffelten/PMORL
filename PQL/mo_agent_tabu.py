from typing import List, Any

from PQL.utils import Reward
from mo_agent import MOGridWorldAgent
from numpy.typing import NDArray
import numpy as np

from mo_env.deep_sea_treasure import DeepSeaTreasure


class MOGridWorldAgentTabu(MOGridWorldAgent):

    def __init__(self, env: DeepSeaTreasure, num_episodes: int, tabu_list_size: int = 100, interactive=False):
        super().__init__(env, num_episodes, interactive=interactive)
        self.tabu_moves: dict[(int, int, int), None] = {}
        self.tabu_list_size = tabu_list_size


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

        non_tabu_actions_values = map(lambda _, action_value: action_value,
                               filter(lambda idx, _: (obs[0], obs[1], idx) not in set(self.tabu_moves.keys()),
                                      enumerate(action_values)))
        biggest_hvs = np.argwhere(action_values == np.amax(non_tabu_actions_values)).flatten()
        if len(biggest_hvs) == 1:
            chosen_action = biggest_hvs[0]
        elif len(biggest_hvs) == 0:  # No non tabu moves
            chosen_action = self.rng.integers(low=0, high=len(action_values))
        else:
            # If there are equalities, randomly chooses among the equal pareto fronts
            chosen_action = biggest_hvs[self.rng.integers(low=0, high=len(biggest_hvs))]

        self.tabu_moves[(obs[0], obs[1], chosen_action)] = None
        # Remove last if tabu table is full
        if len(self.tabu_moves) > self.tabu_list_size:
            print('Removing')
            self.tabu_moves.pop(next(iter(self.tabu_moves)))

        return chosen_action
