# PMORL
Pareto based multi objective reinforcement learning. 
Learns multiple policies in one go. 

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
  * Each file contains for each line `l` the current front seen from the initial state at episode `l` 

## Setup 
1. Create conda environment: `conda env create environment.yml`
2. Activate the env: `conda activate env-MORL`
3. (Change the main if you need)
4. Run `python3 main.py`
