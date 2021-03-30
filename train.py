from gym_env import CombiEnv
from DQNAgent import DQNAgent
import random
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import time
from config import SHOW_PREVIEW, AGGREGATE_STATS_EVERY, EPSILON_DECAY, MIN_EPSILON, EPISODES, MIN_REWARD, MODEL_NAME
from TaskProcessor import TaskList
from CombiApi import api

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

if not os.path.isdir('models'):
    os.makedirs('models')

epsilon = 1
ep_rewards = []

env = CombiEnv(api.size())
agent = DQNAgent(2)

tasklist = TaskList()

for episode in tqdm(range(1, EPISODES +1), ascii=True, unit='episodes'):
    agent.tensorboard.step = episode

    episode_reward = 0
    episode_task_count = 0
    step = 1

    current_state = env.reset()

    done = False

    while not done:
        available_actions = tasklist.get_available(10, env.state['position'], env.state['time'][0]) # Search for eligible actions

        if len(available_actions) == 0:
            # If there are no actions available, idle for one minut
            new_state, reward, done, info = env.step({
                'source': env.state['position'],
                'destination': env.state['position'],
                'time': [60],
                'dist_to_start': [0],
                'mean_dist_to_next': [0]
            })
            current_state = new_state
            continue

        if np.random.random() > epsilon:
            action_index = np.argmax(agent.get_qs(current_state, available_actions))
        else:
            action_index = np.random.randint(0, len(available_actions))

        action = available_actions[action_index]
        action.done = True

        new_state, reward, done, info = env.step(action.to_action_dict())

        episode_reward += reward
        episode_task_count += 1
        
        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, available_actions)
        current_state = new_state
        step += 1

        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
    if len(ep_rewards) > 0:
        prev_max = max(ep_rewards)
    else:
        prev_max = 0

    ep_rewards.append(episode_reward)
    
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon, completed_tasks=episode_task_count)

        if min_reward >= prev_max:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

# Task optimisation: Distance to beginning, number of tasks close to end goal