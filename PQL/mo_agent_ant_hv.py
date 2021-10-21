import utils.softmax
import utils.normmax
from mo_env.deep_sea_treasure import DeepSeaTreasure
from utils import Reward
from mo_agent import MOGridWorldAgent
from numpy.typing import NDArray
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

class MOGridWorldAgentAntHV(MOGridWorldAgent):
    """
    Meta: ACO (repulsive pheromones)
    He: Hypervolume
    """

    def __init__(self,
                 env: DeepSeaTreasure,
                 num_episodes: int,
                 interactive=False,
                 pheromones_decay=0.9,
                 pheromones_weight=1.,
                 output='0',
                 he_weight=1.
                 ):
        super().__init__(env, num_episodes, mode=f'Ant_HV', interactive=interactive, output=output)
        # weight on the heuristic value
        self.he_weight = he_weight
        # weight on the repulsive pheromones
        self.pheromones_weight = pheromones_weight
        self.pheromones = np.ones((self.env.rows, self.env.columns, self.env.actions), dtype=float)
        self.pheromones_decay = pheromones_decay

    def episode_end(self) -> None:
        """
        Updates pheromone levels
        """
        self.pheromones *= self.pheromones_decay

    def step_env(self, obs: NDArray[int], timestep: int) -> tuple[NDArray[int], Reward, bool, int]:
        """
        Overrides step_env to remove the e-greedy meta heuristic
        """
        best_action = self.heuristic(obs)
        self.pheromones[obs[0], obs[1], best_action] += 1

        obs, reward, done = self.env.step(best_action)
        return obs, reward, done, best_action

    def heuristic(self, obs: NDArray[int]) -> int:
        """
            Diversifies by dividing by the number of times we already chose an action in a state
        """
        action_values = self.hv.compute(self.qsets(obs))

        # min clipping, avoid 0s
        action_values[action_values < self.min_val] = self.min_val

        for a in range(len(action_values)):
            # The value of the action are divided by the pheromones (repulsive)
            action_values[a] = (action_values[a] ** self.he_weight) / (self.pheromones[obs[0], obs[1], a] ** self.pheromones_weight)
            if self.nsas[obs[0], obs[1], a] == 0.:
                # Never explored states are super interesting
                action_values[a] = 2**256


        # Using normmax allows to have a probability of choosing other moves than the dominating ones.
        # This is because the pheromone penalty is bounded, meaning there could be actions' hypervolumes
        # which would always be higher than others. Preventing any kind of exploration
        return utils.normmax.normmax(np.array(action_values))

