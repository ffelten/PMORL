import json
import os
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pygmo import hypervolume
import seaborn as sns
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

sns.set_theme(style="darkgrid")

## META PARAMETERS
episodes = 5000
runs = 40
hv_ref_point = np.array([0., 25.])
known_front = [(1., -1.),
               (2., -3.),
               (3., -5.),
               (5., -7.),
               (8., -8.),
               (16., -9.),
               (24., -13.),
               (50., -14.),
               (74., -17.),
               (124., -19.)]
env = "MDST/"
files = os.listdir(env)


def load_front_from_file(filename: str) -> list:
    """
    Reads the content of `filename` and load the front as a list[list[list]]
    First layer is episode, second is for containing all points, third is the number of objectives
    """
    f = open(filename, "r")
    front = []

    for line in f:
        json.loads(line)
        front.append(json.loads(line))

    return front[0:episodes]


def front_hypervolume(front: list, hv_ref_point) -> int:
    """
    Computes hypervolume of the front from the hv_ref_point
    :return: the hypervolume
    """
    # Negates because we maximize and the hv computation supposes minimization
    negated_front = np.array(front) * -1.
    return hypervolume(negated_front).compute(hv_ref_point)


def load_front_hv_for_strategy(files, strategy):
    """
    Loads the front HV per episode in each files matching the current strategy
    :return: a list[list[float]] containing the HV at the end of each episode for each experiment of the given strategy
    """
    all_files_from_strategy = [f for f in files if strategy in f]

    front_hvs_by_run = []

    for strategy_run in all_files_from_strategy:
        front_by_episode = load_front_from_file(env + strategy_run)
        front_hv: list[float] = [front_hypervolume(front, hv_ref_point) for front in front_by_episode]
        front_hvs_by_run.append(front_hv)
    return front_hvs_by_run


def aggregate(hypervolumes: list) -> (list, list):
    """
    Computes the aggregations of the hypervolumes per episode: mean and std dev
    :param hypervolumes: a 2D list of hypervolumes
    :return: two 1D lists, containing the mean hypervolume per episode and the std dev per episode.
    """
    hv_np_arr = np.array(hypervolumes)
    return np.mean(hv_np_arr, axis=0), np.std(hv_np_arr, axis=0)


## COMPUTING DATA

strategies = ["Ant_HV", "Tabu_HV", "E-greedy_HV_fixed", "E-greedy_HV_decaying_episode", "Count_HV"]
# strategies = {"Ant_HV"}
# Dictionary keyed by strategy containing all hypervolumes of all episodes of all runs
# the values are list[list[float]]
front_hvs_by_strategy_by_run = dict()

front_mean_by_strategy = dict()
front_std_by_strategy = dict()
for strategy in strategies:
    front_hvs_by_strategy_by_run[strategy] = load_front_hv_for_strategy(files, strategy)
    front_mean_by_strategy[strategy], front_std_by_strategy[strategy] = aggregate(
        front_hvs_by_strategy_by_run[strategy])

# Better namings for plots
key_mapping = {
    "Ant_HV": "PB",
    "Tabu_HV": "TB",
    "Count_HV": "CB",
    "E-greedy_HV_fixed": "Cε",
    "E-greedy_HV_decaying_episode": "Dε"
}

## PLOTTING

# fig, ax = plt.subplots()
# colors = ['b', 'g', 'r', 'c', 'y']
# plt.xlabel('Episode')
# plt.ylabel('Hypervolume')
# # plt.title('Hypervolume of vectors in start state')
#
# known_front_HV = front_hypervolume(known_front, hv_ref_point)
# ax.plot(range(episodes), [known_front_HV] * episodes, 'k', label='Known Pareto front')


## Add data to plot
# for i, strategy in enumerate(strategies):
#     ci = 1.96 * front_std_by_strategy[strategy] / np.sqrt(runs)
#     ax.plot(range(episodes), front_mean_by_strategy[strategy], colors[i], label=key_mapping[strategy])
#     ax.fill_between(range(episodes), (front_mean_by_strategy[strategy] - ci), (front_mean_by_strategy[strategy] + ci), alpha=.1, color=colors[i])
#
# plt.legend(loc='best', fontsize='small')
# plt.savefig(f'plot_{env[:-1]}', dpi=400)
# plt.show()


## EXACT AGGREGATIONS
#
# mean_hv_every_500_by_strategy = defaultdict(list)
# std_hv_every_500_by_strategy = defaultdict(list)
# for i, strategy in enumerate(strategies):
#     for ep in range(499, episodes, 500):
#         mean_hv_every_500_by_strategy[strategy].append(front_mean_by_strategy[strategy][ep])
#         std_hv_every_500_by_strategy[strategy].append(round(front_std_by_strategy[strategy][ep], 2))
#
# print(mean_hv_every_500_by_strategy)
# print(std_hv_every_500_by_strategy)


## ANOVA test
#
anova_comps: list[list[str]] = [
    ["Ant_HV", "E-greedy_HV_fixed", "E-greedy_HV_decaying_episode"],
    ["Tabu_HV", "E-greedy_HV_fixed", "E-greedy_HV_decaying_episode"],
    ["Count_HV", "E-greedy_HV_fixed", "E-greedy_HV_decaying_episode"]
]
F_500_by_strategy = defaultdict(list)
P_500_by_strategy = defaultdict(list)
Turkey_500_by_strategy = defaultdict(list)  # Turkey test to know which mean is statiscally different in the group
alpha = 0.05
for strategies in anova_comps:
    for ep in range(499, episodes, 500):
        front_hvs: list[list[list[float]]] = [front_hvs_by_strategy_by_run[strategy] for strategy in strategies]
        front_hvs_ep = np.array(front_hvs)[:, :, ep]
        f, p = f_oneway(*front_hvs_ep)
        F_500_by_strategy[strategies[0]].append(f)
        P_500_by_strategy[strategies[0]].append(p)

        # Significantly different means
        if p < alpha:
            df = pd.DataFrame({
                'score': front_hvs_ep.flatten(),
                'group': np.repeat(strategies, repeats=runs)
            })
            turkey = pairwise_tukeyhsd(endog=df['score'], groups=df['group'])
            Turkey_500_by_strategy[strategies[0]].append(turkey)

print(F_500_by_strategy)
print(P_500_by_strategy)
for k, v in Turkey_500_by_strategy.items():
    print(k)
    for res in v:
        print(res)
