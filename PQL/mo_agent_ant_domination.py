from PQL.mo_env.deep_sea_treasure import DeepSeaTreasure
from PQL.utils import Reward
from mo_agent import MOGridWorldAgent
from numpy.typing import NDArray
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

class MOGridWorldAgentAntDomination(MOGridWorldAgent):

    def __init__(self,
                 env: DeepSeaTreasure,
                 num_episodes: int,
                 interactive=False,
                 pheromones_decay=0.9,
                 pheromones_weight=1.,
                 he_weight=0.4
                 ):
        super().__init__(env, num_episodes, mode='ant_domination', interactive=interactive)
        self.he_weight = he_weight
        self.pheromones_weight = pheromones_weight
        self.pheromones = np.zeros((self.env.rows, self.env.columns, self.env.actions), dtype=float)
        self.pheromones_decay = pheromones_decay

    def episode_end(self) -> None:
        """
        Updates pheromone levels
        """
        self.pheromones *= self.pheromones_decay

    def plot_interactive_episode_end(self) -> None:
        super().plot_interactive_episode_end()
        sns.heatmap(self.pheromones.sum(axis=2), linewidth=0.5)
        plt.show()

    def step_env(self, obs: NDArray[int], timestep: int) -> tuple[NDArray[int], Reward, bool, int]:
        """
        Overrides step_env to remove the e-greedy meta heuristic
        """
        best_action = self.heuristic(obs)
        chosen_action = best_action
        self.pheromones[obs[0], obs[1], chosen_action] += 1

        obs, reward, done = self.env.step(chosen_action)
        return obs, reward, done, chosen_action

    def heuristic(self, obs: NDArray[int]) -> int:
        """
            Diversifies by dividing by the number of times we already chose an action in a state
        """
        from utils.domination import moves_containing_nd_points
        action_values = moves_containing_nd_points(self.qsets(obs), self.nd_sets_as_list(obs))

        for a in range(len(action_values)):
            # The value of the action are divided by the pheromones
            # In this case, we increase the values by 1 as to make a false > 0, otherwise unexplored are never chosen
            action_values[a] = (action_values[a] + 1) ** self.he_weight / (self.pheromones[obs[0], obs[1], a] + 1) ** self.pheromones_weight

        best_actions = np.argwhere(action_values == np.amax(action_values)).flatten()
        return np.random.choice(best_actions)

