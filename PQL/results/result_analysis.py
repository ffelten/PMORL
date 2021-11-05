import json
import os

import numpy as np

from pygmo import hypervolume

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

    return front

def front_hypervolume(front: list, hv_ref_point) -> int:
    """
    Computes hypervolume of the front from the hv_ref_point
    :return: the hypervolume
    """
    # Negates because we maximize and the hv computation supposes minimization
    negated_front = np.array(front) * -1.
    return hypervolume(negated_front).compute(hv_ref_point)


hv_ref_point = np.array([0., 25.])
env = "DST/"
files = os.listdir(env)

strategies = {"Ant_HV", "Tabu_HV", "E-greedy_HV_fixed", "E-greedy_HV_decaying_episode"}
# Dictionary keyed by strategy containing all hypervolumes of all episodes of all runs
# the keys are list[list[int]]
front_hvs_by_strategy_by_run = dict()
for strategy in strategies:
    all_files_from_strategy = [f for f in files if strategy in f]

    front_hvs_by_run = []

    for strategy_run in all_files_from_strategy:
        front_by_episode = load_front_from_file(env + strategy_run)
        front_hv = [front_hypervolume(front, hv_ref_point) for front in front_by_episode]
        front_hvs_by_run.append(front_hv)
    front_hvs_by_strategy_by_run[strategy] = front_hvs_by_run

print(front_hvs_by_strategy_by_run.keys())

# Better namings for plots
key_mapping = {
    "Ant_HV": "Pheromones_HV",
    "Tabu_HV": "Tabu_HV",
    "E-greedy_HV_fixed": "Constant_E-greedy_HV",
    "E-greedy_HV_decaying_episode": "Decaying_E-greedy_HV"
}

