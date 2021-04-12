from gym_env import CombiEnv
from DQNAgent import DQNAgent
import random
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import time
from config import SHOW_PREVIEW, MIN_EPSILON, EPISODES
from CombiApi import api
import time
from tests import setups

def train(setup):
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    if not os.path.isdir('models'):
        os.makedirs('models')

    ep_rewards = []

    env = CombiEnv(api.size(), setup)
    agent = DQNAgent(env, 2, setup)


    for episode in tqdm(range(1, EPISODES +1), ascii=True, unit='episodes'):
        agent.tensorboard.step = episode

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

            agent.update_replay_memory((current_state, action, reward, new_state, done))

            agent.train()
            agent.target_train()

            current_state = new_state
            step += 1

            if agent.epsilon > MIN_EPSILON:
                agent.epsilon *= setup['epsilon_decay']
                agent.epsilon = max(MIN_EPSILON, agent.epsilon)

        agent.tensorboard.update_stats(
            episode_reward=episode_reward,
            epsilon=agent.epsilon,
            completed_tasks=extra_stats['completed_tasks'],
            empty_dist=extra_stats['empty_dist']
        )

        if len(ep_rewards) > 0 and episode_reward > max(ep_rewards):
            agent.model.save(f'models/{setup["model_name"]}__{episode_reward}reward__{int(time.time())}.model')

        ep_rewards.append(episode_reward)

    return f"Finished training for {setup['model_name']}"

if __name__ == '__main__':
    config = int(input("What config would you like? "))

    train(setups[config - 1])