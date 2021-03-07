import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def gather_data(env):

    sim_steps = 3000
    num_sims = 5000

    trainX, trainY = [],[]

    for num_sim in range(num_sims):

        observation = env.reset()
        training_sampleX, training_sampleY = [],[]
        success = False

        for step in range(sim_steps):
            # env.render()
            action = np.random.randint(0, 3)
            one_hot_action = np.zeros(3)
            one_hot_action[action] = 1
            # print(one_hot_action)
            training_sampleX.append(observation)
            training_sampleY.append(one_hot_action)

            observation, reward, done, info = env.step(action) 

            if reward != -1:
                trainX += training_sampleX
                trainY += training_sampleY
                success = True
                break
        
        print(num_sim, "/", num_sims, "SUCCESS:",success, "IN STEP:", step)

    trainX, trainY = np.array(trainX), np.array(trainY)

    return trainX, trainY

def create_model():
    model = Sequential()
    model.add(Dense(128, input_shape=(6,), activation="relu"))
    model.add(Dropout(0.6))
    
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.6))
    
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.6))
    
    #added these lines to make model deeper
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.6))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.6))
    
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.6))
    model.add(Dense(3, activation="softmax"))
    
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"])
    
    
    return model


def train():
    env = gym.make("Acrobot-v1")
    trainingX, trainingY = gather_data(env)

    model = create_model()
    model.fit(trainingX, trainingY, epochs = 50)
    
    model.save("Acrobot-v1-4th-try.h5")
    
    return model

if __name__ == '__main__':
    train()
