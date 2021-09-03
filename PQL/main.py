from mo_agent import MOGridWorldAgent
from mo_agent_count_divided import MOGridWorldAgentCountDivided
from mo_agent_qsets import MOGridWorldQSets
from random_agent import RandomAgent
from mo_agent_tabu import MOGridWorldAgentTabu
from mo_env.deep_sea_treasure import DeepSeaTreasure

env = DeepSeaTreasure()

done = False
env.reset()
reward = 0
# mo_env.render()

game = MOGridWorldAgentCountDivided(env, 2000, interactive=True)
# game2 = MOGridWorldQSets(env, 10000, interactive=True)
game.run()
env.reset()
# game2.run()

# while not done:
#     _, reward, done, info = mo_env.step(mo_env.action_space.sample())
#     mo_env.render()
#     print(reward)
#     time.sleep(1)

game.print_end()
# game2.print_end()

print("Voila")
