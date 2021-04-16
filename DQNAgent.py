from CustomTensorBoard import CustomTensorBoard
from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import time

from config import DISCOUNT, REPLAY_MEMORY_SIZE, MINIBATCH_SIZE

from CombiApi import api

class DQNAgent:
    def __init__(self, env, feature_space_size, setup):
        self.env = env

        self.model = self.create_model(feature_space_size, setup['network'])

        self.target_model = self.create_model(feature_space_size, setup['network'])
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = CustomTensorBoard(setup['model_name'], log_dir=f"logs/{setup['model_name']}-{int(time.time())}")

        self.max_dist = api.max_dist()

        self.eligible_actions = []

        self.epsilon = 0.9
        self.tau = 0.125

    def create_model(self, input_dim, layers):
        model = Sequential()

        for i, node_count in enumerate(layers):
            if i > 0:
                model.add(Dense(node_count, activation='relu'))
            else:
                model.add(Dense(node_count, activation='relu', input_dim=input_dim))

        model.add(Dense(1, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=0.001))

        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_features(self, action):
        # Normalise features to converge faster
        features = np.array([
            action.distance_to_start / self.max_dist,
            action.mean_distance_to_next / self.max_dist
        ])
        
        return features.reshape(-1, *features.shape)

    def get_qs(self, state, actions, network='main'):
        features = np.array([self.get_features(action) for action in actions]).reshape(-1,2)

        if network == 'target':
            return self.target_model.predict_on_batch(features)
        
        return self.model.predict_on_batch(features)
    
    def act(self):
        self.eligible_actions = self.env.get_available_actions()

        if len(self.eligible_actions) == 0:
            # If there are no actions available, idle for one minut
            self.env.state['time'][0] += 60
            return False

        if np.random.random() > self.epsilon:
            action_index = np.argmax(self.get_qs(self.env.state, self.eligible_actions))
        else:
            action_index = np.random.randint(0, len(self.eligible_actions))

        action = self.eligible_actions[action_index]
        action.done = True

        return action
    
    def process_minibatch_sample(self, sample):
        (current_state, action, reward, new_current_state, done) = sample

        actions = self.env.get_available_actions(api.find_index(action.destination.id))

        if not done:
            # Compute the maximum Q based on the updated state and the available actions
            # This is very slow - even with just 10 actions
            max_future_q = np.max(self.get_qs(new_current_state, actions, network='target'))
            new_q = reward + DISCOUNT * max_future_q
        else:
            new_q = reward

        features = self.get_features(action)

        return features, new_q
    
    def get_batch_features(self, sample):
        (current_state, action, reward, new_current_state, done) = sample

        actions = self.env.get_available_actions(api.find_index(action.destination.id))

        return [self.get_features(action) for action in actions]
    
    def train(self):
        if len(self.replay_memory) < MINIBATCH_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        X = []
        Y = []

        all_features = []

        number_of_actions = None

        for sample in minibatch:
            features = self.get_batch_features(sample)

            number_of_actions = len(features) if number_of_actions is None else number_of_actions

            all_features.append(features)

        all_features = np.array(all_features).reshape((-1,2))

        targets = self.target_model.predict(all_features)
        
        for i, sample in enumerate(minibatch):
            (current_state, action, reward, new_current_state, done) = sample

            start_index = i * number_of_actions
            end_index = start_index + number_of_actions

            qs = targets[start_index:end_index]

            max_future_q = reward

            if not done:
                max_q = np.max(qs)
                max_future_q += DISCOUNT * max_q

            X.append(self.get_features(action))
            Y.append(max_future_q)

        X = np.array(X).reshape(-1,2)
        Y = np.array(Y)
        self.model.fit(X, Y, batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, epochs=1)
    
    def target_train(self):
        self.target_model.set_weights(self.model.get_weights())
