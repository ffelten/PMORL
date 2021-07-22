
import numpy as np
from gym_mo.envs.gridworlds.mo_gridworld_base import RandomPlayer

class MOAgent(RandomPlayer):
    def __init__(self, env, num_episodes):
        self.env = env
        self.num_episodes = num_episodes

    def run(self):
        episodes = 0
        while episodes < self.num_episodes:
            done = False
            episode_reward = []
            self.env.reset()
            while not done:
                self.env.render()
                r, done = self.step_env()
                episode_reward += r

    def step_env(self):
        _, r, done, _ = self.env.step(self.env.action_space.sample())
        return (r, done)