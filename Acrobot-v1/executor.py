import gym
import time
import numpy as np
from tensorflow import keras

env = gym.make('Acrobot-v1')
# engine = keras.models.load_model('Acrobot-v1-6th-try.h5')
engine = keras.models.load_model('working_models/Acrobot-v1-solved.h5')

observation = env.reset()
done = False
steps = 0

for i in range(2000):
    env.render()
    pred = engine.predict(observation.reshape(1,6))
    action = np.argmax(pred)
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