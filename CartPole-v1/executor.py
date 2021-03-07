from tensorflow import keras
import numpy as np
import gym

engine = keras.models.load_model('CartPole-v1.h5')

done = False

env = gym.make('CartPole-v1')
observation = env.reset()

score = 0
while not done:
    env.render()
    action = np.argmax(engine.predict(
                observation.reshape(1,4)))
    observation, reward, done, _ = env.step(action)
    score += reward

print("DONE, score is:", score)
env.close()


