import gym
import time
import numpy as np

env = gym.make('MountainCar-v0')

observation = env.reset()
done = False
steps = 0

while not done:
    env.render()
    action = env.action_space.sample()

    position = observation[0]
    velocity = observation[1]

    if position > 0 and velocity > 0:
        action = 2
    elif position < 0 and velocity < 0:
        action = 0
    else:
        action = 2
    
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