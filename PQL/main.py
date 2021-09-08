from PQL.mo_agent_qset_tabu import MOGridWorldQSetsTabu
from mo_agent import MOGridWorldAgent
from mo_agent_count_divided import MOGridWorldAgentCountDivided
from mo_agent_qsets import MOGridWorldQSets
from random_agent import RandomAgent
from mo_agent_tabu import MOGridWorldAgentTabu
from mo_env.deep_sea_treasure import DeepSeaTreasure
from mo_agent_domination import MOGridWorldAgentDomination
from matplotlib import pyplot as plt

env = DeepSeaTreasure()

done = False
env.reset()
reward = 0
# mo_env.render()

games = [
    MOGridWorldAgent(env, 3500, interactive=False),
    MOGridWorldAgentDomination(env, 3500, interactive=False),
    # MOGridWorldAgentCountDivided(env, 1000, interactive=False),
    # MOGridWorldAgentTabu(env, 1000, interactive=False, tabu_list_size=100),
    MOGridWorldQSets(env, 3500, interactive=False),
    # MOGridWorldQSetsTabu(env, 1000, interactive=False, tabu_list_size=100),
]

plt.ylim([0, 10])
plt.xlabel('Episodes')
plt.ylabel('Found points on the front')
plt.title('Front over time using different heuristics')

def add_plot(front_over_time, mode):
    plt.plot(front_over_time, label=mode)

for game in games:
    env.reset()
    game.run()
    add_plot(game.found_points_episodes, game.mode)
    game.print_end()

plt.legend()
plt.show()
print("Voila")
