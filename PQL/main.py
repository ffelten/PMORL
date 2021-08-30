from mo_agent import MOGridWorldAgent
from mo_agent_diversity import MOGridWorldAgentDiversity
from mo_env.deep_sea_treasure import DeepSeaTreasure

env = DeepSeaTreasure()

done = False
env.reset()
reward = 0
# mo_env.render()

game = MOGridWorldAgent(env, 2000, interactive=False)
game2 = MOGridWorldAgentDiversity(env, 2000, interactive=False)
game.run()
env.reset()
game2.run()

# while not done:
#     _, reward, done, info = mo_env.step(mo_env.action_space.sample())
#     mo_env.render()
#     print(reward)
#     time.sleep(1)

game.print_end()
game2.print_end()

print("Voila")
