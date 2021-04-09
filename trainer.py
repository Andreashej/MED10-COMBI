from multiprocessing import Process
from train import train

setups = [
    { # Default config for comparison
        'model_name': 'Config1',
        'network': [12, 24, 12],
        'speed': 2,
        'epsilon_decay': 0.99975,
        'reward': None
    },
    { # Low nodecount in the network
        'model_name': 'Config2',
        'network': [3, 6, 3],
        'speed': 2,
        'epsilon_decay': 0.99975,
        'reward': None,
    },
    { # Moves faster
        'model_name': 'Config3',
        'network': [12, 24, 12],
        'speed': 8,
        'epsilon_decay': 0.99975,
        'reward': None
    },
    { # Try with a small positive reward for completing the action
        'model_name': 'Config4',
        'network': [12, 24, 12],
        'speed': 2,
        'epsilon_decay': 0.99975,
        'reward': lambda action : 100 - (action['dist_to_start'][0] + action['mean_dist_to_next'][0])
    },
    { # Give a smaller weight to mean_dist_to_next and favor tasks closer to the worker
        'model_name': 'Config5',
        'network': [12, 24, 12],
        'speed': 2,
        'epsilon_decay': 0.99975,
        'reward': lambda action : (action['dist_to_start'][0] + 0.5 * action['mean_dist_to_next'][0])
    },
    { # Slower decay
        'model_name': 'Config6',
        'network': [12, 24, 12],
        'speed': 2,
        'epsilon_decay': 0.99999,
        'reward': None
    },
]

if __name__ == '__main__':
    processes = []

    for setup in setups:
        process = Process(target=train, args=(setup,))
        process.start()