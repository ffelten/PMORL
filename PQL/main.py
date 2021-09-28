from PQL.mo_agent_qset_tabu import MOGridWorldQSetsTabu
from PQL.mo_env.pyramid import Pyramid
from mo_agent import MOGridWorldAgent
from mo_agent_count_divided import MOGridWorldAgentCountDivided
from mo_agent_qsets import MOGridWorldQSets
from random_agent import RandomAgent
from mo_agent_tabu import MOGridWorldAgentTabu
from mo_env.deep_sea_treasure import DeepSeaTreasure
from mo_agent_domination import MOGridWorldAgentDomination
from mo_agent_ant_hv import MOGridWorldAgentAntHV
from mo_agent_ant_domination import MOGridWorldAgentAntDomination
from matplotlib import pyplot as plt

env = Pyramid()

done = False
env.reset()
reward = 0
# mo_env.render()
eps = 1000

games = [
    MOGridWorldAgent(env, eps, interactive=False),
    MOGridWorldAgentAntHV(env, eps, interactive=False, pheromones_decay=0.95, he_weight=0.5, pheromones_weight=1.),
    # MOGridWorldAgentAntDomination(env, eps, interactive=False, pheromones_decay=0.95, he_weight=0.4, pheromones_weight=1.),
    # RandomAgent(env, eps, interactive=False),
    # MOGridWorldAgentDomination(env, eps, interactive=False),
    MOGridWorldAgentCountDivided(env, eps, interactive=False),
    # MOGridWorldAgentTabu(env, eps, interactive=False, tabu_list_size=100),
    # MOGridWorldQSets(env, eps, interactive=False),
    # MOGridWorldQSetsTabu(env, eps, interactive=False, tabu_list_size=100),
]

# plt.ylim([0, 10])
plt.xlabel('Episodes')
plt.ylabel('Found points on the front')
plt.title('Front over time using different heuristics')

def add_plot(front_over_time, mode):
    plt.plot(front_over_time, label=mode)

for game in games:
    env.reset()
    game.run()
    add_plot(game.found_points_episodes, game.mode)
    # game.print_end()

plt.legend()
plt.show()
print("Voila")

games[0].plot_interactive_episode_end()
games[1].plot_interactive_episode_end()
games[2].plot_interactive_episode_end()
