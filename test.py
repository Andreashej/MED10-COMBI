from numpy.lib.function_base import average
from gym_env import CombiEnv
from DQNAgent import DQNAgent
import random
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import time
from config import SHOW_PREVIEW, MIN_EPSILON, TEST_EPISODES, UPDATE_TARGET_EVERY
from CombiApi import api
import time
from tests import setups

def test(setup):
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    if not os.path.isdir('models'):
        os.makedirs('models')

    ep_rewards = []
    ep_completed_tasks = []
    ep_empty_dist = []
    
    setup['speed'] = 6

    env = CombiEnv(api.size(), setup)
    agent = DQNAgent(env, 2, setup, 'Config5__101307.0reward__1618210028.model')


    for episode in tqdm(range(1, TEST_EPISODES +1), ascii=True, unit='episodes'):
        episode_reward = 0
        extra_stats = {
            'completed_tasks': 0,
            'empty_dist': 0
        }
        step = 1

        current_state = env.reset()

        done = False

        while not done:
            action = agent.act()

            if not action:
                continue
            
            new_state, reward, done, info = env.step(action.to_action_dict())

            extra_stats['completed_tasks'] += 1
            extra_stats['empty_dist'] += info['empty_dist']

            episode_reward += reward
            
            if SHOW_PREVIEW:
                env.render()

            current_state = new_state
            step += 1

        ep_rewards.append(episode_reward)
        ep_completed_tasks.append(extra_stats['completed_tasks'])
        ep_empty_dist.append(extra_stats['empty_dist'])
    
    print(f"Completed tasks: {average(ep_completed_tasks)}")
    print(f"Empty distance: {average(ep_empty_dist)}")

if __name__ == '__main__':
    config = int(input("What config would you like? "))

    test(setups[config - 1])