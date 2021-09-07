import numpy as np
from numpy.typing import NDArray

from PQL.mo_agent import MOGridWorldAgent

import pygmo as pg

from PQL.mo_env.deep_sea_treasure import DeepSeaTreasure


class MOGridWorldAgentDomination(MOGridWorldAgent):

    def __init__(self, env: DeepSeaTreasure, num_episodes: int, interactive=False):
        super().__init__(env, num_episodes, mode='e_greedy_dominance', interactive=interactive)

    def heuristic(self, obs: NDArray[int]) -> int:
        """
            Checks if the qset contains one non dominated point on the front.
            Tie break randomly for non dominated qsets
        :param obs:
        :return:
        """
        qsets = self.qsets(obs)
        nd_set = list(self.non_dominated_sets(obs).to_set())  # list will keep the order
        negated_qset = list(map(lambda arr: [-arr[0], -arr[1]], nd_set))

        if (len(negated_qset)) <= 1:
            return self.env.sample_action()

        ndf = [nd_set[p] for p in pg.fast_non_dominated_sorting(points=negated_qset)[0][0]]
        front = set(ndf)

        non_dominated_moves = np.zeros_like(qsets)

        for a in range(len(qsets)):
            if len(qsets[a].to_set().intersection(front)) > 0:
                non_dominated_moves[a] = True

        non_dominated_moves = np.argwhere(non_dominated_moves == 1).flatten()
        if len(non_dominated_moves) > 1:
            return non_dominated_moves[self.rng.integers(low=0, high=len(non_dominated_moves))]
        else:
            return non_dominated_moves[0]
