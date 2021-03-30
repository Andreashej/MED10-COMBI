from CustomTensorBoard import CustomTensorBoard
from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import time

from config import DISCOUNT, REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, UPDATE_TARGET_EVERY, MODEL_NAME

from CombiApi import api

class DQNAgent:
    def __init__(self, feature_space_size):
        self.model = self.create_model(feature_space_size)

        self.target_model = self.create_model(feature_space_size)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = CustomTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0

        self.max_dist = api.max_dist()

    def create_model(self, input_dim):
        model = Sequential()

        model.add(Dense(12, activation='relu', input_dim=input_dim))
        # model.add(Dropout(0.2))
        
        model.add(Dense(24, activation='relu'))
        # model.add(Dropout(0.2))

        model.add(Dense(12, activation='relu'))
        # model.add(Dropout(0.2))

        model.add(Dense(1, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=0.001))

        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_features(self, state, action):
        # Normalise features to converge faster
        features = np.array([
            action.distance_to_start / self.max_dist,
            action.mean_distance_to_next / self.max_dist
        ])
        
        return features.reshape(-1, *features.shape)

    def get_qs(self, state, actions):
        return np.array([self.get_q(state, action) for action in actions])
    
    def get_q(self, state, action):
        features = self.get_features(state, action)

        return self.model.predict(features)
    
    def train(self, terminal_state, actions):
        if len(self.replay_memory) < MINIBATCH_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # future_qs_list = np.array([self.get_q(transition[3], transition[1]) for transition in minibatch])

        X = []
        Y = []

        for (current_state, action, reward, new_current_state, done) in minibatch:
            if not done:
                # Compute the maximum Q based on the updated state and the available actions
                # This is very slow - even with just 10 actions
                max_future_q = np.max(self.get_qs(new_current_state, actions))
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            X.append(self.get_features(current_state, action))
            Y.append(new_q)

        X = np.array(X).reshape((-1,2))
        Y = np.array(Y)

        self.model.fit(X, Y, batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, epochs=1)

        if terminal_state:
            self.target_update_counter += 1
        
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0