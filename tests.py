EPSILON_DECAY = 0.99985

setups = [
    { # Default config for comparison
        'model_name': 'Config1',
        'network': [12, 24, 12],
        'speed': 2,
        'epsilon_decay': EPSILON_DECAY,
        'I': 0,
        'alpha': 1,
        'beta': 1
    },
    { # Low nodecount in the network
        'model_name': 'Config2',
        'network': [3, 6, 3],
        'speed': 2,
        'epsilon_decay': EPSILON_DECAY,
        'I': 0,
        'alpha': 1,
        'beta': 1
    },
    { # Deeper network
        'model_name': 'Config3',
        'network': [12, 24, 6, 24, 12],
        'speed': 2,
        'epsilon_decay': EPSILON_DECAY,
        'I': 0,
        'alpha': 1,
        'beta': 1
    },
    { # Higher nodecount
        'model_name': 'Config4',
        'network': [32, 64, 32],
        'speed': 2,
        'epsilon_decay': EPSILON_DECAY,
        'I': 0,
        'alpha': 1,
        'beta': 1
    },
    { # Try with a small positive reward for completing the action
        'model_name': 'Config5',
        'network': [3, 6, 3],
        'speed': 2,
        'epsilon_decay': EPSILON_DECAY,
        'I': 100,
        'alpha': 1,
        'beta': 1
    },
    { # Give a smaller weight to mean_dist_to_next and favor tasks closer to the worker
        'model_name': 'Config6',
        'network': [3, 6, 3],
        'speed': 2,
        'epsilon_decay': EPSILON_DECAY,
        'I': 0,
        'alpha': 1,
        'beta': 0.5
    },
    { # Large positive reward for finishing
        'model_name': 'Config7',
        'network': [3, 6, 3],
        'speed': 2,
        'epsilon_decay': EPSILON_DECAY,
        'I': 1000,
        'alpha': 1,
        'beta': 1
    },
    { # Smaller weight on agent - task distance
        'model_name': 'Config8',
        'network': [3, 6, 3],
        'speed': 2,
        'epsilon_decay': EPSILON_DECAY,
        'I': 0,
        'alpha': 0.5,
        'beta': 1
    },
]