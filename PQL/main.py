from mo_agent import MOGridWorldAgent
from mo_env.deep_sea_treasure import DeepSeaTreasure

env = DeepSeaTreasure()

done = False
env.reset()
reward = 0
# mo_env.render()

game = MOGridWorldAgent(env, 10000, interactive=False)
game.run()

# while not done:
#     _, reward, done, info = mo_env.step(mo_env.action_space.sample())
#     mo_env.render()
#     print(reward)
#     time.sleep(1)

game.print_end()

print("Voila")
