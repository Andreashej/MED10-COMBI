import gym
from gym import spaces
import numpy as np
import random

IMMEDIATE_AWARD = 100 # Reward for completing a task
ALPHA = 1 # Blend factor for how much travelling empty is penalised
EPISODE_LENGTH = 36000 # Seconds

class CombiEnv(gym.Env):

    metadata = { 'render.modes': ['human'] }

    def __init__(self, bin_count, shift_length=36000):
        super(CombiEnv, self).__init__()

        self.observation_space = spaces.Dict({ 'position': spaces.Discrete(bin_count), 'time': spaces.Box(np.array([0]), np.array([shift_length]), None, np.uint16) })
        self.action_space = spaces.Dict({ 
            'source': spaces.Discrete(bin_count), 
            'destination': spaces.Discrete(bin_count),
            'time': spaces.Box(np.array([60]), np.array([3600]), None, np.uint16),
            'vicinity': spaces.Box(np.array([0]), np.array([100]), None, np.uint16)
        })
        self.done = False

        self.reset()

    def step(self, action):
        empty_dist = self.bin_dist(self.state['position'], action['source'])
        task_dist = self.bin_dist(action['source'], action['destination'])

        self.state['position'] = action['destination']
        self.state['time'][0] = self.state['time'][0] + action['time'][0]

        reward = IMMEDIATE_AWARD * task_dist - ALPHA * self.bin_dist(self.state['position'], action['source']) + action['vicinity'][0]

        if self.state['time'][0] > EPISODE_LENGTH:
            self.done = True

        return self.state, reward, self.done, { 'empty_dist': empty_dist }

    def reset(self):
        self.state = {
            'position': self.observation_space.sample()['position'],
            'time': [0]
        }
        self.done = False

        return self.state

    def render(self, mode='human', close=False):
        print (f"I am now at BIN {self.state['position']}. The time is {self.state['time'][0 ]}.")

    @staticmethod
    def bin_dist(source, destination):
        return random.randint(0,100)
    
    @staticmethod
    def tasks_in_radius(bin, radius, tasks):
        taskcount = 0
        for task in tasks:
            dist = CombiEnv.bin_dist(bin, task['destination'])
            if dist <= radius:
                taskcount += 1
        
        return taskcount
    
    def available_actions(self):
        return [self.action_space.sample() for _ in range(0,10)]
