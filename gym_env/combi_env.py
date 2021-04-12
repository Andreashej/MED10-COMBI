from threading import current_thread
import gym
from gym import spaces
import numpy as np
from CombiApi import api
from TaskProcessor import TaskList

class CombiEnv(gym.Env):

    metadata = { 'render.modes': ['human'] }

    def __init__(self, bin_count, setup, shift_length=36000):
        super(CombiEnv, self).__init__()

        self.observation_space = spaces.Dict({ 'position': spaces.Discrete(bin_count), 'time': spaces.Box(np.array([0]), np.array([shift_length]), None, np.uint16) })
        self.action_space = spaces.Dict({ 
            'source': spaces.Discrete(bin_count), 
            'destination': spaces.Discrete(bin_count),
            'time': spaces.Box(np.array([60]), np.array([3600]), None, np.uint16),
            'dist_to_start': spaces.Box(np.array([0]), np.array([api.size()]), None, np.uint16),
            'mean_dist_to_next': spaces.Box(np.array([0]), np.array([api.size()]), None, np.uint16)
        })
        self.done = False
        self.tasklist = TaskList()

        self.EPISODE_LENGTH = shift_length # Seconds - defaults to 36000 = 10 hours
        self.SPEED = setup['speed'] # Used to estimate the time it takes to move to the start of a task, unit is BIN_DIST/s
        self.REWARD = setup['reward'] # Pass in a custom reward function that takes the action as input

        self.last_minute = 0

        self.reset()

    def step(self, action):
        self.state['position'] = action['destination']
        self.state['time'][0] = self.state['time'][0] + action['time'][0] + ((action['dist_to_start'][0] + 1) / self.SPEED)
        
        current_minute = (self.state['time'][0] / 60) % 30

        if current_minute < self.last_minute:
            self.tasklist.spawn(self.state['time'][0], 30)
            self.last_minute = current_minute

        if not self.REWARD:
            reward = (action['dist_to_start'][0] + action['mean_dist_to_next'][0]) * -1
        else:
            reward = self.REWARD(action)

        if self.state['time'][0] > self.EPISODE_LENGTH:
            self.done = True

        return self.state, reward, self.done, { 'empty_dist': action['dist_to_start'][0] }

    def reset(self):
        self.state = {
            'position': self.observation_space.sample()['position'],
            'time': [0]
        }
        self.done = False
        self.last_minute = 0
        self.tasklist.reset()

        return self.state

    def render(self, mode='human', close=False):
        print (f"I am now at BIN {self.state['position']}. The time is {self.state['time'][0 ]}.")

    def get_available_actions(self, position = None, time = None):
        if not position:
            position = self.state['position']
        
        if not time:
            time = self.state['time'][0]

        return self.tasklist.get_available(position, time)
