from gym_mo.envs.gridworlds import MODeepSeaTresureEnv
import time

my_grid = MODeepSeaTresureEnv(from_pixels=True)

done = False
my_grid.reset()
while not done:
    _, r, done, _ = my_grid.step(my_grid.action_space.sample())
    my_grid.render()
    time.sleep(0.5)

print("Voila")
