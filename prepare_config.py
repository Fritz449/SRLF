import json

config_name = 'rainbow.json'
args = \
    {
        "learning_rate": 0.0001,
        'config': config_name,
        "xp_size": 100000,
        "test_every": 20000,
        "scale": False,
        "batch_size": 64,
        "tau": 0.001,
        "n_pre_tasks": 4,
        "n_atoms": 101,
        "n_workers": 4,
        "double": False,
        "n_steps": 1,
        "env_name": "LunarLander-v2",
        "n_tests": 12,
        "gamma": 0.99,
        "dueling": True,
        "prioritized": True,
        "prior_alpha": 0.5,
        "prior_beta": 0.6,
        "continuous": False,
        "max_pathlength": 2000,
        "random_steps": 10000,
        "env_type": "gym",
        "n_hiddens": [
            128, 64
        ],
        "noisy_nn": False,
        "factorized_noise": True,
        "trainer": "Rainbow",
        "max_magnitude": 100.0
    }
print (args)
with open('configs/' + config_name, 'w') as outfile:
    json.dump(args, outfile, indent=4)
