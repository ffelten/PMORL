import numpy as np

from PQL.utils.QSet import QSet
from PQL.utils.hv_indicator import MaxHVHeuristic
from mo_agent_qset_tabu import MOGridWorldQSetsTabu
from mo_env.hard_deep_sea_treasure import HardDeepSeaTreasure
from mo_env.bonus_world import BonusWorld
from mo_env.pyramid import Pyramid
from mo_agent import MOGridWorldAgent
from mo_agent_count_divided import MOGridWorldAgentCountDivided
from mo_agent_hv_count_divided import MOGridWorldAgentHVCountDivided
from mo_agent_qsets import MOGridWorldQSets
from random_agent import RandomAgent
from mo_agent_tabu import MOGridWorldAgentTabu
from mo_env.deep_sea_treasure import DeepSeaTreasure
from mo_agent_domination import MOGridWorldAgentDomination
from mo_agent_ant_hv import MOGridWorldAgentAntHV
from mo_agent_ant_domination import MOGridWorldAgentAntDomination
from matplotlib import pyplot as plt

env = BonusWorld()

done = False
env.reset()
reward = 0
# mo_env.render()
eps = 5000
interactive = False
runs = 25

## Shape is (runs, games)
games = [[
    # MOGridWorldAgent(env, eps, interactive=interactive, output=str(r)),
    # MOGridWorldAgentAntHV(env, eps, interactive=interactive, pheromones_decay=0.9, he_weight=1., pheromones_weight=2., output=str(r)),
    # MOGridWorldAgentAntDomination(env, eps, interactive=interactive, pheromones_decay=0.8, he_weight=1., pheromones_weight=2.),
    # RandomAgent(env, eps, interactive=interactive),
    # MOGridWorldAgentDomination(env, eps, interactive=interactive),
    # MOGridWorldAgentCountDivided(env, eps, interactive=interactive),
    # MOGridWorldAgentHVCountDivided(env, eps, interactive=interactive, count_weight=3, output=str(r)),
    MOGridWorldAgentTabu(env, eps, interactive=interactive, tabu_list_size=150, output=str(r)),
    # MOGridWorldQSets(env, eps, interactive=interactive),
    # MOGridWorldQSetsTabu(env, eps, interactive=interactive, tabu_list_size=100),
] for r in range(runs)]


def add_plot(front_over_time, mode):
    plt.plot(front_over_time, label=mode)


for r in range(runs):
    print(f'Run {r}/{runs}')
    for i, game in enumerate(games[r]):
        env.reset()
        game.run()
        # add_plot(game.hv_over_time, game.mode)
    # game.print_end()

## Shape is (run, game, episode)
hypervolumes = np.array([[g.hv_over_time for g in games[r]] for r in range(runs)]).reshape((runs, len(games[0]), eps))

print(np.mean(hypervolumes, axis=0))
hvs = np.mean(hypervolumes, axis=0)
ci = 1.96 * np.std(hypervolumes, axis=0) / np.sqrt(runs)

fig, ax = plt.subplots()
colors = ['b', 'g', 'r', 'c']
for i, g in enumerate(games[0]):
    ax.plot(range(eps), hvs[i], colors[i], label=g.mode)
    ax.fill_between(range(eps), (hvs[i] - ci[i]), (hvs[i] + ci[i]), alpha=.1, color=colors[i])

front_HV = MaxHVHeuristic(env.hv_point).compute(np.array([QSet(env.front)]))
ax.plot(range(eps), [front_HV] * eps, 'k', label='Front')

plt.xlabel('Episodes')
plt.ylabel('Hypervolume')
plt.title('HV of the front from initial state over time')
plt.legend()
plt.show()
print("Voila")

games[0].plot_interactive_episode_end()
games[1].plot_interactive_episode_end()
games[2].plot_interactive_episode_end()
