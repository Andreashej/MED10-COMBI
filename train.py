from gym_env import CombiEnv
from DQNAgent import DQNAgent
import random
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import time
from config import SHOW_PREVIEW, AGGREGATE_STATS_EVERY, EPSILON_DECAY, MIN_EPSILON, EPISODES, MIN_REWARD, MODEL_NAME

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

if not os.path.isdir('models'):
    os.makedirs('models')

epsilon = 1
ep_rewards = [MIN_REWARD]

env = CombiEnv(500)
agent = DQNAgent(env)

def dynamicActionSearch():
    return [env.action_space.sample() for _ in range(0,10)]


for episode in tqdm(range(1, EPISODES +1), ascii=True, unit='episodes'):
    agent.tensorboard.step = episode

    episode_reward = 0
    step = 1

    current_state = env.reset()

    done = False

    while not done:
        available_actions = dynamicActionSearch() # Search for eligible actions

        if np.random.random() > epsilon:
            action_index = np.argmax(agent.get_qs(current_state, available_actions))
        else:
            action_index = np.random.randint(0, len(available_actions))

        action = available_actions[action_index]

        new_state, reward, done, info = env.step(action)

        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)
        current_state = new_state
        step += 1

        ep_rewards.append(episode_reward)
        
        if not episode_reward % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            if min_reward >= MIN_REWARD:
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')


        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

# Task optimisation: Distance to beginning, number of tasks close to end goal