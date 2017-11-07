import json

config_name = 'rainbow.json'
args = \
    {
        'trainer': 'Rainbow',
        'config': config_name,
        'continuous': False,
        'env_type': 'gym',
        'env_name': 'CartPole-v0',
        'max_pathlength': 250,
        'n_workers': 4,
        'xp_size': 20000,
        'batch_size': 64,
        'n_pre_tasks': 4,
        'n_hiddens': [64],
        'n_tests': 10,
        'n_atoms': 51,
        'max_magnitude': 20.,
        'tau': 0.001,
        'test_every': 5000,
        'random_steps': 10000,
        'learning_rate': 0.0001,
        'gamma': 0.9,
        'scale': False
    }
print (args)
with open('configs/' + config_name, 'w') as outfile:
    json.dump(args, outfile, indent=4)
