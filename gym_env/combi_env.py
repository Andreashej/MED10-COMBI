import gym
from gym import spaces
import numpy as np
import random
from CombiApi import api

IMMEDIATE_AWARD = 100 # Reward for completing a task
ALPHA = 1 # Blend factor for how much travelling empty is penalised
EPISODE_LENGTH = 36000 # Seconds
SPEED = 2 # Used to estimate the time it takes to move to the start of a task, unit is BIN_DIST/s

class CombiEnv(gym.Env):

    metadata = { 'render.modes': ['human'] }

    def __init__(self, bin_count, shift_length=36000):
        super(CombiEnv, self).__init__()

        self.observation_space = spaces.Dict({ 'position': spaces.Discrete(bin_count), 'time': spaces.Box(np.array([0]), np.array([shift_length]), None, np.uint16) })
        self.action_space = spaces.Dict({ 
            'source': spaces.Discrete(bin_count), 
            'destination': spaces.Discrete(bin_count),
            'time': spaces.Box(np.array([60]), np.array([3600]), None, np.uint16),
            'dist_to_start': spaces.Box(np.array([0]), np.array([8000]), None, np.uint16),
            'mean_dist_to_next': spaces.Box(np.array([0]), np.array([8000]), None, np.uint16)
        })
        self.done = False

        self.reset()

    def step(self, action):
        self.state['position'] = action['destination']
        self.state['time'][0] = self.state['time'][0] + action['time'][0] + ((action['dist_to_start'][0] + 1) / SPEED)

        reward = (action['dist_to_start'][0] + action['mean_dist_to_next'][0]) * -1

        if self.state['time'][0] > EPISODE_LENGTH:
            self.done = True

        return self.state, reward, self.done, { 'empty_dist': action['dist_to_start'][0] }

    def reset(self):
        self.state = {
            'position': self.observation_space.sample()['position'],
            'time': [0]
        }
        self.done = False

        return self.state

    def render(self, mode='human', close=False):
        print (f"I am now at BIN {self.state['position']}. The time is {self.state['time'][0 ]}.")
