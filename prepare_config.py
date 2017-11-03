import json

config_name = 'ddpg_lland.json'
args = \
    {
        'trainer': 'DDPG',
        'config': config_name,
        'continuous': True,
        'env_type': 'gym',
        'env_name': 'LunarLanderContinuous-v2',
        'max_pathlength': 2000,
        'n_workers': 4,
        'xp_size': 20000,
        'batch_size': 64,
        'n_pre_tasks': 4,
        'n_hiddens': [400, 300],
        'n_tests': 10,
        'action_noise': 0.0,
        'tau': 0.001,
        'step_delay': 0.01,
        'test_every': 50000,
        'random_steps': 10000,
        'learning_rate': 0.0001,
        'learning_rate_critic': 0.001,
        'gamma': 0.99,
        'scale': False
    }
print (args)
with open('configs/' + config_name, 'w') as outfile:
    json.dump(args, outfile, indent=4)
