import numpy as np

from mo_env.deep_sea_treasure import DeepSeaTreasure, Viewport


class BonusWorld(DeepSeaTreasure):

    def __init__(self):
        # the map
        # -1,-1 is the x2 bonus
        # -2,-2 are the pits
        self.map = np.array(
            [[(0, 0),   (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (1, 9)],
             [(0, 0),   (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (-2, -2),   (0, 0)],
             [(0, 0),   (0, 0),     (-10, -10), (-10, -10), (0, 0),     (0, 0),     (0, 0),     (0, 0),     (3, 9)],
             [(0, 0),   (0, 0),     (-10, -10), (-1, -1),   (0, 0),     (0, 0),     (0, 0),     (-2, -2),   (0, 0)],
             [(0, 0),   (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (5, 9)],
             [(0, 0),   (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (-2, -2),   (0, 0)],
             [(0, 0),   (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (7, 9)],
             [(0, 0),   (-2, -2),   (0, 0),     (-2, -2),   (0, 0),     (-2, -2),   (0, 0),     (0, 0),     (0, 0)],
             [(9, 1),   (0, 0),     (9, 3),     (0, 0),     (9, 5),     (0, 0),     (9, 7),     (0, 0),     (9, 9)]]
        )

        self.sea_map = self.map

        self.front = {
            (1, 9, -8), (3, 9, -10), (5, 9, -12), (10, 18, -14), (14, 18, -16), (18, 18, -18),
            (18, 12, -16), (18, 10, -14), (9, 5, -12), (9, 3, -10), (9, 1, -8)
        }

        # DON'T normalize
        self.max_reward = 1.0

        self.rng = np.random.default_rng(42)

        self.rows = self.map.shape[0]
        self.columns = self.map.shape[1]

        # state space specification: 2-dimensional discrete box
        self.state_spec = [['discrete', 1, [0, self.rows - 1]], ['discrete', 1, [0, self.columns - 1]]]

        # first, second, time reward
        self.reward_spec = [[0, 18], [0, 18], [-1, 0]]

        # action space specification: 1 dimension, 0 up, 1 down, 2 left, 3 right
        self.actions = 4
        self.action_spec = ['discrete', 1, [0, self.actions - 1]]

        self.hv_point = np.array([0., 0., 50.])

        # Starts bottom left corner
        self.current_state = np.array([0, 0])
        self.init_state = self.current_state
        self.terminal = False
        self.bonus = False

        ## Rendering
        self.viewport = Viewport()
        self.setup_viewer_configuration()

    def reset(self):
        '''
        reset the location of the submarine
        '''
        self.current_state = self.init_state
        self.terminal = False
        self.bonus = False
        return self.current_state

    def step(self, action):
        '''
        step one move and feed back reward
        '''
        dirs = {
            0: np.array([-1, 0]),  # up
            1: np.array([1, 0]),  # down
            2: np.array([0, -1]),  # left
            3: np.array([0, 1])  # right
        }

        dir_for_action = dirs[action]

        next_state = self.current_state + dir_for_action

        in_grid = lambda x, ind: \
            (x[ind] >= self.state_spec[ind][2][0]) and \
            (x[ind] <= self.state_spec[ind][2][1])

        if in_grid(next_state, 0) and in_grid(next_state, 1):
            if (self.get_map_value(next_state) != [-10, -10]).all():
                self.current_state = next_state
            # pit
            if np.array_equal(self.get_map_value(next_state), (-2, -2)):
                self.current_state = self.init_state


        (obj1, obj2) = self.get_map_value(self.current_state)
        # empty cell
        if obj1 == 0 and obj2 == 0:
            reward = (0, 0, -1)
        # x2
        elif obj1 == -1 and obj2 == -1:
            self.bonus = True
            reward = (0, 0, -1)
        else:
            if self.bonus:
                reward = (2*obj1, 2*obj2, -1)
            else:
                reward = (obj1, obj2, -1)

            self.terminal = True


        return self.current_state, reward, self.terminal

    def get_map_value(self, pos) -> (float, float):
        return self.map[pos[0]][pos[1]]

    def value_to_color(self, value):
        if value.sum() == 0:
            return 255.0, 255.0, 255.0
        elif value.sum() > 0:
            return 255.0, 255.0, 0.
        elif value.sum() == -2:
            return 0., 255., 0.
        elif value.sum() == -4:
            return 255., 0., 0.
        else:
            return 0., 0., 0.