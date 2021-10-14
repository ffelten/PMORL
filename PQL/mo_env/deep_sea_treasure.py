from __future__ import absolute_import, division, print_function

import random

import numpy as np
from gym.envs.classic_control import rendering

# Logic part was copied from https://github.com/RunzheYang/MORL/blob/master/synthetic/envs/deep_sea_treasure.py
# Then modified according to the needs of the research
#
# Renderring part was copied and adapted from https://github.com/johan-kallstrom/gym-mo/blob/master/gym_mo/envs/gridworlds/gridworld_base.py
class DeepSeaTreasure(object):

    def __init__(self, sea_map = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-10, 2, 0, 0, 0, 0, 0, 0, 0, 0],
             [-10, -10, 3, 0, 0, 0, 0, 0, 0, 0],
             [-10, -10, -10, 5, 8, 16, 0, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, 24, 50, 0, 0],
             [-10, -10, -10, -10, -10, -10, -10, -10, 0, 0],
             [-10, -10, -10, -10, -10, -10, -10, -10, 74, 0],
             [-10, -10, -10, -10, -10, -10, -10, -10, -10, 124]]
        )):
        # the map of the deep sea treasure
        self.sea_map = sea_map

        self.front = {
            (1., -1.),
            (2., -3.),
            (3., -5.),
            (5., -7.),
            (8., -8.),
            (16., -9.),
            (24., -13.),
            (50., -14.),
            (74., -17.),
            (124., -19.)
        }

        self.hv_point = np.array([0., 25.])

        # DON'T normalize
        self.max_reward = 1.0

        self.rng = np.random.default_rng(42)

        self.rows = self.sea_map.shape[0]
        self.columns = self.sea_map.shape[1]

        # state space specification: 2-dimensional discrete box
        self.state_spec = [['discrete', 1, [0, self.rows - 1]], ['discrete', 1, [0, self.columns - 1]]]

        # action space specification: 1 dimension, 0 up, 1 down, 2 left, 3 right
        self.actions = 4
        self.action_spec = ['discrete', 1, [0, self.actions - 1]]

        # reward specification: 2-dimensional reward
        # 1st: treasure value || 2nd: time penalty
        self.reward_spec = [[0, 124], [-1, 0]]

        self.init_state = np.array([0, 0])
        self.current_state = self.init_state
        self.terminal = False

        ## Rendering
        self.viewport = Viewport()
        self.setup_viewer_configuration()

    def get_map_value(self, pos):
        return self.sea_map[pos[0]][pos[1]]

    def reset(self):
        '''
            reset the location of the submarine
        '''
        self.current_state = self.init_state
        self.terminal = False
        return self.current_state

    def step(self, action):
        '''
            step one move and feed back reward
        '''
        dir = {
            0: np.array([-1, 0]),  # up
            1: np.array([1, 0]),  # down
            2: np.array([0, -1]),  # left
            3: np.array([0, 1])  # right
        }[action]
        next_state = self.current_state + dir

        in_grid = lambda x, ind: \
            (x[ind] >= self.state_spec[ind][2][0]) and \
            (x[ind] <= self.state_spec[ind][2][1])


        if in_grid(next_state, 0) and in_grid(next_state, 1):
            if self.get_map_value(next_state) != -10:
                self.current_state = next_state

        treasure_value = self.get_map_value(self.current_state)
        if treasure_value == 0 or treasure_value == -1:
            treasure_value = 0.0
        else:
            treasure_value /= self.max_reward
            self.terminal = True
        time_penalty = -1.0 / self.max_reward
        reward = np.array([treasure_value, time_penalty])

        return self.current_state, reward, self.terminal

    def sample_action(self):
        return self.rng.integers(low=0, high=self.actions)

    ## RENDERING
    def setup_viewer_configuration(self):
        self.SCALE_H = self.viewport.height / (self.rows + 2)
        self.SCALE_W = self.viewport.width / (self.columns + 2)
        self.viewer = None
        self.on_key_press = None
        self.on_key_release = None
        self.render_grid = True

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.viewport.width, self.viewport.height)
            self.viewer.set_bounds(0, self.viewport.width / self.SCALE_W, 0, self.viewport.height / self.SCALE_H)
            if (self.on_key_press is not None) and (self.on_key_release is not None):
                self.viewer.window.on_key_press = self.on_key_press
                self.viewer.window.on_key_release = self.on_key_release

        # Draw background
        self.viewer.draw_polygon([(0, 0),
                                  (self.viewport.width / self.SCALE_W, 0),
                                  (self.viewport.width / self.SCALE_W, self.viewport.height / self.SCALE_H),
                                  (0, self.viewport.height / self.SCALE_H),
                                  ], color=(0.0, 0.0, 0.0))

        # Draw grid border
        self.viewer.draw_polyline([(1, 1),
                                   (1, self.rows + 1),
                                   (self.columns + 1, self.rows + 1),
                                   (self.columns + 1, 1),
                                   (1, 1),
                                   ], color=(255.0, 255.0, 255.0))

        # Draw grid lines
        if self.render_grid:
            for i in range(2, self.columns + 1):
                self.viewer.draw_polyline([(i, 1), (i, self.rows + 1)], color=(255.0, 255.0, 255.0))
            for i in range(2, self.rows + 1):
                self.viewer.draw_polyline([(1, i), (self.columns + 1, i)], color=(255.0, 255.0, 255.0))

        # Draw objects
        for row in range(self.sea_map.shape[0]):
            for column in range(self.sea_map.shape[1]):
                value = self.sea_map[row, column]
                self.render_object(column + 1, self.rows - row, self.value_to_color(value))

        # Draw grid agent
        # (!!!) do self.rows - to reverse the rep of the y axis (y=0 is bottom for viewport, but it top for matrix view)
        self.render_object(self.current_state[1] + 1, self.rows - self.current_state[0], (0., 0., 255.))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def value_to_color(self, value):
        if value == 0:
            return 255.0, 255.0, 255.0
        elif value > 0:
            return 255.0, 255.0, 0.
        else:
            return 0., 0., 0.


    def render_object(self, x, y, object_color):
        x1 = x + self.viewport.object_delta
        x2 = x - self.viewport.object_delta + 1
        y1 = y + self.viewport.object_delta
        y2 = y - self.viewport.object_delta + 1
        self.viewer.draw_polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), ], color=object_color)


class Viewport:
    """Class for viewport settings.

    """

    def __init__(self, viewport_width=600, viewport_height=600, view_port_object_delta=0.1):
        self.width = viewport_width
        self.height = viewport_height
        self.object_delta = view_port_object_delta
