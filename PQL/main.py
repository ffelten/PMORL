import time
from env.deep_sea_treasure import DeepSeaTreasure
from mo_agent import MOGridWorldAgent

env = DeepSeaTreasure()

done = False
env.reset()
reward = 0
# env.render()

game = MOGridWorldAgent(env, 10000, interactive=False)
game.run()

# while not done:
#     _, reward, done, info = env.step(env.action_space.sample())
#     env.render()
#     print(reward)
#     time.sleep(1)

game.print_end()

print("Voila")
