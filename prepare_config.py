import json

config_name = 'es_d_lland.json'
args = \
    {
        'trainer': 'ES',
        'config': config_name,
        'continuous': False,
        'env_type': 'gym',
        'env_name': 'LunarLander-v2',
        'max_pathlength': 1000,
        'noise_scale': 0.3,
        'n_workers': 4,
        'n_tasks': 256,
        'n_pre_tasks': 4,
        'n_hiddens': [512],
        'n_tests': 10,
        'learning_rate': 0.01,
        'momentum': True,
        'std': 'Const',
        'ranks': True,
        'scale': True
    }
print (args)
with open('configs/' + config_name, 'w') as outfile:
    json.dump(args, outfile, indent=4)
