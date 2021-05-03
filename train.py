from gym_env import CombiEnv
from DQNAgent import DQNAgent
import random
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import time
from config import SHOW_PREVIEW, MIN_EPSILON, EPISODES, UPDATE_TARGET_EVERY, AGGREGATE_STATS_EVERY
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
    ep_completed_tasks = []
    ep_empty_dist = []

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
            step_start = time.perf_counter()
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

            if step % UPDATE_TARGET_EVERY == 0:
                agent.target_train()

            current_state = new_state
            step += 1

            if agent.epsilon > MIN_EPSILON:
                agent.epsilon *= setup['epsilon_decay']
                agent.epsilon = max(MIN_EPSILON, agent.epsilon)
            step_end = time.perf_counter()

            # print(f"Step completed in {step_end - step_start} seconds")

        ep_rewards.append(episode_reward)
        ep_completed_tasks.append(extra_stats['completed_tasks'])
        ep_empty_dist.append(extra_stats['empty_dist'])

        if episode % AGGREGATE_STATS_EVERY == 0 or episode == 1:
            if episode == 1:
                mean_reward = ep_rewards[0]
                min_reward = mean_reward
                max_reward = mean_reward
                mean_episode_completed = ep_completed_tasks[0]
                mean_empty_dist = ep_empty_dist[0]
            else:
                episode_rewards = ep_rewards[episode - AGGREGATE_STATS_EVERY : episode]
                mean_reward = sum(episode_rewards) / len(episode_rewards)
                min_reward = min(episode_rewards)
                max_reward = max(episode_rewards)

                episode_completed = ep_completed_tasks[episode - AGGREGATE_STATS_EVERY : episode]
                mean_episode_completed = sum(episode_completed) / len(episode_completed)

                episode_empty_dist = ep_empty_dist[episode - AGGREGATE_STATS_EVERY : episode]
                mean_empty_dist = sum(episode_empty_dist) / len(episode_empty_dist)

            agent.tensorboard.update_stats(
                reward=mean_reward,
                min_reward=min_reward,
                max_reward=max_reward,
                completed_tasks=mean_episode_completed,
                empty_distance=mean_empty_dist
            )

        if len(ep_rewards) > 0 and episode_reward > max(ep_rewards):
            agent.model.save(f'models/{setup["model_name"]}__{episode_reward}reward__{int(time.time())}.model')

    return f"Finished training for {setup['model_name']}"

if __name__ == '__main__':
    config = int(input("What config would you like? "))

    train(setups[config - 1])