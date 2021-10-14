from __future__ import absolute_import, division, print_function

import random

import numpy as np
from gym.envs.classic_control import rendering

from mo_env.deep_sea_treasure import DeepSeaTreasure


class HardDeepSeaTreasure(DeepSeaTreasure):

    def __init__(self):
        # the map of the deep sea treasure
        sea_map = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                       1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, -10,                     -10, 2, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, -10, -10,                   -10, -10, 3, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, -10, -10, -10,                 -10, -10, -10, 5, 8, 16, 0, 0, 0, 0],
             [0, 0, 0, 0, -10, -10, -10, -10, -10, -10,           -10, -10, -10, -10, -10, -10, 0, 0, 0, 0],
             [0, 0, 0, 0, -10, -10, -10, -10, -10, -10,           -10, -10, -10, -10, -10, -10, 0, 0, 0, 0],
             [0, 0, 0, 0, -10, -10, -10, -10, -10, -10,           -10, -10, -10, -10, -10, -10, 24, 50, 0, 0],
             [0, 0, -10, -10, -10, -10, -10, -10, -10, -10,       -10, -10, -10, -10, -10, -10, -10, -10, 0, 0],
             [0, 0, -10, -10, -10, -10, -10, -10, -10, -10,       -10, -10, -10, -10, -10, -10, -10, -10, 74, 0],
             [0, -10, -10, -10, -10, -10, -10, -10, -10, -10,     -10, -10, -10, -10, -10, -10, -10, -10, -10, 124]]
        )
        super().__init__(sea_map)
        self.hv_point = np.array([0., 55.])
        self.init_state = np.array([0, 10])
        self.current_state = self.init_state
