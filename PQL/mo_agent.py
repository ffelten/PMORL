from itertools import product

import numpy as np
from numpy.typing import NDArray

from mo_env.deep_sea_treasure import DeepSeaTreasure
from utils import Reward
from utils.QSet import QSet
from matplotlib import pyplot as plt
from utils.hv_indicator import MaxHVHeuristic


class MOGridWorldAgent:
    """
    Multi objective Pareto based RL
    """

    def __init__(
            self,
            env: DeepSeaTreasure,
            num_episodes: int,
            mode: str = 'e_greedy_HV',
            num_objectives: int = 2,
            alpha=0.01,
            gamma=1,
            interactive=True,
            epsilon=0.997,
            policies=10,
            seed=42
    ):
        self.env = env
        self.num_episodes = num_episodes
        self.num_objectives = num_objectives
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.mode = mode
        self.policies = policies

        self.hv = MaxHVHeuristic(np.array([0., 25.]))
        # q set for each state-action pair
        self.nd_set = np.empty((self.env.rows, self.env.columns, self.env.actions), dtype=object)
        for i, j, a in product(range(self.env.rows), range(self.env.columns), range(self.env.actions)):
            self.nd_set[i, j, a] = QSet([])

        # the number of objectives is not necessarily the shape of the reward from the mo_env!
        self.avg_rewards: NDArray[float] = np.zeros((self.env.rows, self.env.columns, self.env.actions, num_objectives))
        # number of times we chose state action pair
        self.nsas: NDArray[int] = np.zeros((self.env.rows, self.env.columns, self.env.actions), dtype=int)

        self.interactive = interactive

        self.rng = np.random.default_rng(seed)

        self.found_points_episodes = []

    def run(self):
        """
        Core loop, runs the agent on the environment for a number of episode.
        """
        episode = 0
        while episode < self.num_episodes:
            done = False
            obs = self.env.reset()
            timestep = 0
            self.initial_state = obs
            if self.interactive:
                self.env.render()

            while not done and timestep < 1000:
                # Move
                next_obs, r, done, a = self.step_env(obs, timestep)
                # Learn
                self.update_rewards(obs, a, r)
                self.update_NDset(obs, a, next_obs)

                # Iterate
                obs = next_obs
                timestep += 1
                if done:
                    self.print_episode_end(episode)
                    episode += 1

                if self.interactive:
                    self.env.render()
                #     time.sleep(0.5)

            # if episode % 500 == 0:
            #     self.print_end()

    ### UPDATES ###

    def update_rewards(self, obs: NDArray[int], action: int, reward: Reward) -> None:
        """
        Update the avg reward vector for obs, action using reward_vector
        :param obs: the observation at previous timestep
        :param action: the action taken at previous timestep
        :param reward: the reward obtained after taking action
        """
        self.nsas[obs[0], obs[1], action] += 1
        rsa = self.avg_rewards[obs[0], obs[1], action]
        self.avg_rewards[obs[0], obs[1], action] = rsa + (reward - rsa) / self.nsas[obs[0], obs[1], action]

    def update_NDset(self, obs: NDArray[int], action: int, next_obs: NDArray[int]) -> None:
        """
        Updates the NDset of obs,action pair from the ones seen in next_obs
        :param obs: old observation
        :param action: action taken at obs
        :param next_obs: new observation
        """
        self.nd_set[obs[0], obs[1], action] = self.non_dominated_sets(next_obs)
        # Prunes the nd set
        self.nd_set[obs[0], obs[1], action].shrink(self.policies)

    ### MOVES ###

    def step_env(self, obs: NDArray[int], timestep: int) -> tuple[NDArray[int], Reward, bool, int]:
        """
        Chooses one action and performs it
        :param obs: current obs
        :param timestep: current timestep
        :return: next obs, reward, whether done or not, action performed
        """
        best_action = self.heuristic(obs)
        chosen_action = self.e_greedy(best_action, self.exploration_proba(timestep))

        obs, reward, done = self.env.step(chosen_action)
        return obs, reward, done, chosen_action

    ### MOVES CHOICE ###

    def exploration_proba(self, timestep: int) -> float:
        """
        Gives the proba for exploration, by default it is an exponential decay: epsilon ^ timestep
        :param timestep: current timestep
        :return: the exploration probability for this timestep
        """
        return self.epsilon ** timestep

    def qsets(self, obs: NDArray[int]) -> NDArray[QSet]:
        """
        Computes the qsets for each action starting current obs.
        Applies TD update on a clone to avoid modifying true NDSets
        """
        current_nd_sets = self.nd_set[obs[0], obs[1]]
        qsets = np.empty_like(current_nd_sets)
        for a in range(len(current_nd_sets)):
            qsets[a] = current_nd_sets[a].clone_td(self.gamma, self.avg_rewards[obs[0], obs[1], a])
        return qsets

    def heuristic(self, obs: NDArray[int]) -> int:
        """
        Heuristic to choose the next action from obs
        :param obs: current obs
        :return: the next best action
        """
        action_values = self.hv.compute(self.qsets(obs))

        biggest_hvs = np.argwhere(action_values == np.amax(action_values)).flatten()
        if len(biggest_hvs) == 1:
            return biggest_hvs[0]
        else:
            # If there are equalities, randomly chooses among the equal pareto fronts
            return biggest_hvs[self.rng.integers(low=0, high=len(biggest_hvs))]

    def e_greedy(self, best_action: int, epsilon: float) -> int:
        """
        Epsilon-greedy - chooses a random action with proba epsilon, and best_action with proba (1-epsilon)
        :param epsilon: probability of choosing a random move
        :param best_action: the action to choose if the coin turns to choose best
        :return: the index of the chosen action
        """
        coin = self.rng.random()
        if coin < epsilon:
            return self.env.sample_action()
        else:
            return best_action

    ### UTILS ###
    def print_episode_end(self, ep) -> None:
        front = self.get_init_state_front()
        found_points_this_episode = len(front.to_set().intersection(self.env.front))
        self.found_points_episodes.append(found_points_this_episode)

        print("########################")
        print("Episode %s done" % ep)
        print("Episode front: %s" % front)
        print("Found points on the front: %s" % found_points_this_episode)
        print("########################")

    def non_dominated_sets(self, obs: NDArray[int]) -> QSet:
        """
        Computes and return the set of non dominated reward vectors from obs over actions (supposed minimization)
        :param obs: the current observation
        :return: ND(U_{a \in actions} Qsets(obs,a))
        """
        union = QSet()
        qsets = self.qsets(obs)
        for a in range(self.env.actions):
            union.append(qsets[a])
        return union

    def get_init_state_front(self) -> QSet:
        tmp = self.non_dominated_sets(self.initial_state)
        tmp.shrink(self.policies)
        return tmp

    def print_end(self):
        front = self.get_init_state_front()
        front.draw_front_2d()
        print("Final front: %s " % front)