import gym
import time
import numpy as np

env = gym.make('Acrobot-v1')

observation = env.reset()
done = False
steps = 0

for i in range(2000):
    env.render()
    action = env.action_space.sample()
   
    observation, reward, done, info = env.step(action)

    print("STEP:", steps)
    print("ACTION:", action)
    print("OBSERVATION:",observation)
    print("REWARD:",reward)
    print("DONE:",done)
    print("INFO:",info)
    print("-----------------------------------")

    steps += 1
    if reward != -1:
        break

print(steps)
env.close()