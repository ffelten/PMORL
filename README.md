# PMORL
Pareto based multi objective reinforcement learning. 
Learns multiple policies in one go.

## Linked paper

```
@conference{icaart22,
author={Florian Felten. and Grégoire Danoy. and El{-}Ghazali Talbi. and Pascal Bouvry.},
title={Metaheuristics-based Exploration Strategies for Multi-Objective Reinforcement Learning},
booktitle={Proceedings of the 14th International Conference on Agents and Artificial Intelligence - Volume 2: ICAART,},
year={2022},
pages={662-673},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0010989100003116},
isbn={978-989-758-547-0},
}
```
[https://www.scitepress.org/Link.aspx?doi=10.5220/0010989100003116]

## Learning algorithm
Implementation is based on the paper: K. Van Moffaert and A. Nowé, “Multi-objective reinforcement learning using sets of pareto dominating policies,” The Journal of Machine Learning Research, vol. 15, no. 1, pp. 3483–3512, 2014.

## Heuristics
* Hypervolume
* Pareto domination

## Meta-heuristics
* Ant colony based
* Count based
* Tabu based
* e-greedy
* Random search

## Files
* ``mo_agent.py`` contains the implementation of the learning and a basic e-greedy exploration strategy.
* ``mo_agent_*.py`` contain other implementations of meta heuristics/heuristics combination.
* ``utils/`` contains some utilities shared between implementations
* ``mo_env/`` contains the implementations of some multi objective grid world
* ``results/`` contains files with the results of some of our experiments.
  * File names follow the format `HyperHeuristic_Heuristic_DATE_TIME` (except some variants of e-greedy that I messed up):
    * `E-greedy_HV` -> epsilon greedy hypervolume, decaying exponentially on timestep base: 0.997^timestep.
    * `E-greedy_HV_decaying_episode` -> epsilon greedy hypervolume, decaying exponentially on episode base: 0.997^episode
    * `E-greedy_HV_fixed` -> epsilon greedy hypervolume, fixed to 0.4 random proba.
  * Each file contains for each line `l` the current front seen from the initial state at the end of episode `l`.

## Setup 
1. Create conda environment: `conda env create environment.yml`
2. Activate the env: `conda activate env-MORL`
3. (Change the main if you need)
4. Run `python3 main.py`
