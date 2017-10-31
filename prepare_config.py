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
        'timesteps_batch': 10000,
        'noise_scale': 0.3,
        'n_workers': 4,
        'xp_size': 20000,
        'batch_size': 128,
        'n_pre_tasks': 4,
        'n_hiddens': [128, 128],
        'n_tests': 10,
        'learning_rate': 0.01,
        'max_kl': 0.01,
        'gamma': 0.995,
        'momentum': True,
        'std': 'Param',
        'ranks': False,
        'scale': True
    }
print (args)
with open('configs/' + config_name, 'w') as outfile:
    json.dump(args, outfile, indent=4)
