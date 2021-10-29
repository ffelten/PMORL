import json
import pickle
from itertools import product

import numpy as np
from numpy.typing import NDArray

import utils.argmax
from mo_env.deep_sea_treasure import DeepSeaTreasure
from utils import Reward
from utils.QSet import QSet
from matplotlib import pyplot as plt
from datetime import datetime
import matplotlib.pylab as plt2
import seaborn as sns
from utils.hv_indicator import MaxHVHeuristic


class MOGridWorldAgent:
    """
    Multi objective Pareto based RL
    Behavioral policy follows epsilon greedy hypervolume by default
    """

    def __init__(
            self,
            env: DeepSeaTreasure,
            num_episodes: int,
            mode: str = 'E-greedy_HV',
            output: str = '0',
            gamma=1,
            interactive=True,
            epsilon=0.997,
            policies=100
    ):
        self.env = env
        self.num_episodes = num_episodes
        self.num_objectives = len(self.env.reward_spec)
        self.gamma = gamma
        self.epsilon = epsilon
        # string describing the current mode
        self.mode = mode
        # to define file where to write results for later analysis
        self.output = output
        now = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        self.output_file = open("results/" + self.mode + "_" + output + "_" + now + ".json", "w")
        # max number of policies per state action QSet
        self.policies = policies

        self.hv = MaxHVHeuristic(env.hv_point)
        # min clipping value
        self.min_val = 1.
        # non dominated set for each state-action pair
        self.nd_sets = np.empty((self.env.rows, self.env.columns, self.env.actions), dtype=object)
        for i, j, a in product(range(self.env.rows), range(self.env.columns), range(self.env.actions)):
            self.nd_sets[i, j, a] = QSet([])

        # the number of objectives is not necessarily the shape of the reward from the mo_env!
        self.avg_rewards: NDArray[float] = np.zeros(
            (self.env.rows, self.env.columns, self.env.actions, self.num_objectives))
        # number of times we chose state action pair
        self.nsas: NDArray[int] = np.zeros((self.env.rows, self.env.columns, self.env.actions), dtype=int)

        # boolean to render or not
        self.interactive = interactive

        # performance analysis
        self.found_points_episodes = []
        self.hv_over_time = []
        self.front_over_time = []

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

            # optimization to avoid continuing to learn once we have the full front
            if self.get_init_state_front().to_set() == self.env.front:
                done = True
                self.gather_data()
                episode += 1

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
                    self.episode_end()
                    self.print_episode_end(episode)
                    episode += 1

                # Render env
                if self.interactive:
                    self.env.render()

            if episode % 10 == 0 and self.interactive:
                for p in self.get_init_state_front().set:
                    r = self.track(p)
                    print(f'Reward for tracking {p} = {r}')
                # self.plot_interactive_episode_end()
            #     self.print_end()
        self.output_file.close()

    ### UPDATES ###

    def episode_end(self) -> None:
        """
        Hook that is called when the episode ends, can be overriden
        """
        pass

    def update_rewards(self, obs: NDArray[int], action: int, reward: Reward) -> None:
        """
        Update the avg reward vector for obs, action using reward
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
        self.nd_sets[obs[0], obs[1], action] = self.non_dominated_sets(next_obs)
        # Prunes the nd set
        self.nd_sets[obs[0], obs[1], action].shrink(self.policies)

    ### MOVES ###

    def step_env(self, obs: NDArray[int], timestep: int) -> tuple[NDArray[int], Reward, bool, int]:
        """
        Chooses one action and performs it
        - epsilon greedy as meta
        :param obs: current obs
        :param timestep: current timestep (!) normally in RL, it should be episode instead of timestep
        :return: next obs, reward, whether done or not, action performed
        """
        best_action = self.heuristic(obs)
        chosen_action = self.e_greedy(best_action, self.exploration_proba(timestep))

        obs, reward, done = self.env.step(chosen_action)
        return obs, reward, done, chosen_action

    ### MOVES CHOICE ###

    def exploration_proba(self, timestep: int) -> float:
        """
        Gives the proba for exploration, by default it follows an exponential decay: epsilon ^ timestep
        :param timestep: current timestep
        :return: the exploration probability for this timestep
        """
        return self.epsilon ** timestep

    def qsets(self, obs: NDArray[int]) -> NDArray[QSet]:
        """
        Computes the qsets for each action starting current obs.
        QSet(s,a) = NDSet(s,a) (+) R(s,a)
        Applies TD update on a clone to avoid modifying true NDSets
        """
        current_nd_sets = self.nd_sets[obs[0], obs[1]]
        qsets = np.empty_like(current_nd_sets)
        for a, nd_set in enumerate(current_nd_sets):
            qsets[a] = nd_set.clone_td(self.gamma, self.avg_rewards[obs[0], obs[1], a])
        return qsets

    def heuristic(self, obs: NDArray[int]) -> int:
        """
        Heuristic to choose the next action from obs
         hypervolume by default
        :param obs: current obs
        :return: the next best action
        """
        action_values = self.hv.compute(self.qsets(obs))

        return utils.argmax.argmax(action_values)

    def e_greedy(self, best_action: int, epsilon: float) -> int:
        """
        Epsilon-greedy - chooses a random action with proba epsilon, and best_action with proba (1-epsilon)
        :param epsilon: probability of choosing a random move
        :param best_action: the action to choose if the coin turns to choose best
        :return: the index of the chosen action
        """
        coin = np.random.rand()
        if coin < epsilon:
            return np.random.choice(range(self.env.actions))
        else:
            return best_action

    ### UTILS ###
    def plot_interactive_episode_end(self) -> None:
        """
        Plots a heatmap of where the agent spent his time
        """
        sns.heatmap(self.nsas.sum(axis=2), linewidth=0.5)
        plt2.show()

    def gather_data(self):
        front = self.get_init_state_front()
        self.output_file.write(json.dumps([s.tolist() for s in front.set]) + '\n')

        found_points_this_episode = len(front.to_set().intersection(self.env.front))
        self.front_over_time.append(front)
        hv = MaxHVHeuristic(self.env.hv_point)
        self.hv_over_time.append(hv.compute(np.array([front])))
        self.found_points_episodes.append(found_points_this_episode)
        return front, found_points_this_episode

    def print_episode_end(self, ep) -> None:
        front, found_points_this_episode = self.gather_data()

        print("########################")
        print(f"Episode {ep} done, mode={self.mode}")
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

    def nd_sets_as_list(self, obs) -> list:
        """ Clones the nd_set as to avoid mutations """
        return list(self.non_dominated_sets(obs).to_set())  # list will keep the order

    def get_init_state_front(self) -> QSet:
        tmp = self.non_dominated_sets(self.initial_state)
        tmp.shrink(self.policies)
        return tmp

    def print_end(self):
        front = self.get_init_state_front()
        front.draw_front_2d()
        print("Final front: %s " % front)

    def track(self, target):
        """
        Tracking a target point using the current knowledge
        :param target: the target point (a point in the objective space)
        :return: the reward accumulated by following a policy targetting the target point
        """
        print(f'Tracking {target}')
        done = False
        obs = self.env.reset()
        self.env.render()
        reward = np.array([0.] * self.num_objectives)
        while not done:
            action = np.random.randint(0, self.env.actions)
            for a in range(self.env.actions):
                rsa = self.avg_rewards[obs[0], obs[1], a]
                ndset = self.nd_sets[obs[0], obs[1], a]
                for qvec in ndset.set:
                    if np.array_equal(self.gamma * qvec + rsa, target):
                        action = a
                        target = qvec

            obs, r, done = self.env.step(action)
            reward += r
            self.env.render()
        return reward
