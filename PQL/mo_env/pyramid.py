import numpy as np

from PQL.mo_env.deep_sea_treasure import DeepSeaTreasure, Viewport


class Pyramid(DeepSeaTreasure):

    def __init__(self):
        # the map
        self.map = np.array(
            [[(0, 100), (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10)],
             [(0, 0),   (10, 90),   (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10)],
             [(0, 0),   (0, 0),     (20, 80),   (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10)],
             [(0, 0),   (0, 0),     (0, 0),     (30, 70),   (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10)],
             [(0, 0),   (0, 0),     (0, 0),     (0, 0),     (40, 60),   (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10)],
             [(0, 0),   (0, 0),     (0, 0),     (0, 0),     (0, 0),     (50, 50),   (-10, -10), (-10, -10), (-10, -10), (-10, -10), (-10, -10)],
             [(0, 0),   (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (60, 40),   (-10, -10), (-10, -10), (-10, -10), (-10, -10)],
             [(0, 0),   (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (70, 30),   (-10, -10), (-10, -10), (-10, -10)],
             [(0, 0),   (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (80, 20),   (-10, -10), (-10, -10)],
             [(0, 0),   (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (90, 10),   (-10, -10)],
             [(0, 0),   (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (0, 0),     (100, 0)]]
        )

        self.sea_map = self.map

        self.front = {
            (-9., 91), (1., 81.), (11., 71.), (21., 61.), (31., 51.),
            (41., 41.), (51., 31.), (61., 21.), (71., 11.), (81., 1.),
            (91., -9)
        }

        # DON'T normalize
        self.max_reward = 1.0

        self.rng = np.random.default_rng(42)

        self.rows = self.map.shape[0]
        self.columns = self.map.shape[1]

        # state space specification: 2-dimensional discrete box
        self.state_spec = [['discrete', 1, [0, self.rows - 1]], ['discrete', 1, [0, self.columns - 1]]]

        # action space specification: 1 dimension, 0 up, 1 down, 2 left, 3 right
        self.actions = 4
        self.action_spec = ['discrete', 1, [0, self.actions - 1]]

        # reward specification: 2-dimensional reward
        self.reward_spec = [[-1, 100], [-1, 100]]
        self.hv_point = np.array([20.1, 20.1])

        # Starts bottom left corner
        self.current_state = np.array([self.rows - 1, 0])
        self.terminal = False

        ## Rendering
        self.viewport = Viewport()
        self.setup_viewer_configuration()

    def reset(self):
        '''
        reset the location of the submarine
        '''
        self.current_state = np.array([self.rows - 1, 0])
        self.terminal = False
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

        # stochastic transitions
        # coin = np.random.random_sample()
        # if coin < 0.95:
        next_state = self.current_state + dir_for_action
        # else:
        #     next_state = self.current_state + dirs[np.random.randint(low=0, high=4)]

        in_grid = lambda x, ind: \
            (x[ind] >= self.state_spec[ind][2][0]) and \
            (x[ind] <= self.state_spec[ind][2][1])

        if in_grid(next_state, 0) and in_grid(next_state, 1):
            if (self.get_map_value(next_state) != [-10, -10]).all():
                self.current_state = next_state

        (obj1, obj2) = self.get_map_value(self.current_state)
        if obj1 == 0 and obj2 == 0:
            reward = (np.random.normal(-1, 0.0), np.random.normal(-1, 0.0))
        else:
            reward = (np.random.normal(obj1, 0.01), np.random.normal(obj2, 0.01))
            self.terminal = True


        return self.current_state, reward, self.terminal

    def get_map_value(self, pos) -> (float, float):
        return self.map[pos[0]][pos[1]]

    def value_to_color(self, value):
        if value.sum() == 0:
            return 255.0, 255.0, 255.0
        elif value.sum() > 0:
            return 255.0, 255.0, 0.
        else:
            return 0., 0., 0.