from CombiApi import api
from config import DISCOUNT
import numpy as np

max_dist = api.max_dist()

def get_features(action):
    # Normalise features to converge faster
    features = np.array([
        action.distance_to_start / max_dist,
        action.mean_distance_to_next / max_dist
    ])
    
    return features.reshape(-1, *features.shape)

def get_batch_features(args):
    (sample, env) = args

    (current_state, action, reward, new_current_state, done) = sample

    actions = env.get_available_actions(api.find_index(action.destination.id))

    return [get_features(action) for action in actions]