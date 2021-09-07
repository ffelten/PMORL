from PQL.mo_agent_qset_tabu import MOGridWorldQSetsTabu
from mo_agent import MOGridWorldAgent
from mo_agent_count_divided import MOGridWorldAgentCountDivided
from mo_agent_qsets import MOGridWorldQSets
from random_agent import RandomAgent
from mo_agent_tabu import MOGridWorldAgentTabu
from mo_env.deep_sea_treasure import DeepSeaTreasure
from mo_agent_domination import MOGridWorldAgentDomination

env = DeepSeaTreasure()

done = False
env.reset()
reward = 0
# mo_env.render()

games = [
    MOGridWorldAgent(env, 1000, interactive=False),
    # MOGridWorldAgentCountDivided(env, 1000, interactive=False),
    # MOGridWorldAgentTabu(env, 1000, interactive=False, tabu_list_size=100),
    # MOGridWorldQSets(env, 1000, interactive=False),
    # MOGridWorldQSetsTabu(env, 1000, interactive=False, tabu_list_size=100),
]

for game in games:
    env.reset()
    game.run()
    game.print_end()

print("Voila")
