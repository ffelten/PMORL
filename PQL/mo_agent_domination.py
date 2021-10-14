import numpy as np
from numpy.typing import NDArray

import utils.argmax
from mo_agent import MOGridWorldAgent

import pygmo as pg

from mo_env.deep_sea_treasure import DeepSeaTreasure


class MOGridWorldAgentDomination(MOGridWorldAgent):

    def __init__(self, env: DeepSeaTreasure, num_episodes: int, interactive=False):
        super().__init__(env, num_episodes, mode='e_greedy_domination', interactive=interactive)

    def heuristic(self, obs: NDArray[int]) -> int:
        """
            Checks if the qset contains one non dominated point on the front.
            Tie break randomly for non dominated qsets
        :param obs:
        :return:
        """
        qsets = self.qsets(obs)
        nd_set = self.nd_sets_as_list(obs)
        from utils.domination import moves_containing_nd_points
        non_dominated_moves = moves_containing_nd_points(qsets, nd_set)

        return PQL.utils.argmax.argmax(non_dominated_moves)
