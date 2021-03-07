import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def gather_data(env):

    min_score = 50
    sim_steps = 500
    num_sims = 10000

    trainX, trainY = [],[]
    scores = []

    for _ in range(num_sims):

        observation = env.reset()
        score = 0
        training_sampleX, training_sampleY = [],[]

        for step in range(sim_steps):
            # env.render()
            action = np.random.randint(0, 2)

            one_hot_action = np.zeros(2)
            one_hot_action[action] = 1

            training_sampleX.append(observation)
            training_sampleY.append(one_hot_action)

            observation, reward, done, info = env.step(action) 
            score +=  reward

            if done:
                break

        if score > min_score:
            scores.append(score)
            trainX += training_sampleX
            trainY += training_sampleY

    trainX, trainY = np.array(trainX), np.array(trainY)
    print("Average: {}".format(np.mean(scores)))
    print("Median: {}".format(np.median(scores)))
    return trainX, trainY

def create_model():
    model = Sequential()
    model.add(Dense(128, input_shape=(4,), activation="relu"))
    model.add(Dropout(0.6))
    
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.6))
    
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.6))
    
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.6))
    
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.6))
    model.add(Dense(2, activation="softmax"))
    
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"])
    
    
    return model


def train():
    env = gym.make("CartPole-v1")
    trainingX, trainingY = gather_data(env)

    model = create_model()
    model.fit(trainingX, trainingY, epochs = 5)
    
    model.save("CartPole-v1.h5")
    
    return model

if __name__ == '__main__':
    train()
