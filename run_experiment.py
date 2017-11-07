import json
import os
import sys
import helpers.utils as hlp
import tensorflow as tf

sys.path.append(os.path.realpath(".."))
config_name = 'rainbow.json'

with open('configs/' + config_name, 'r') as fp:
    config = json.load(fp)
algo = hlp.agent_from_config(config)
env = hlp.env_from_config(config)
config['environment'] = env
sess = tf.InteractiveSession()
config['n_features'] = env.get_observation_space()
config['n_actions'] = env.get_action_space()
agent = algo(sess, config)
#agent.load(config_name[:-5])
agent.train()
