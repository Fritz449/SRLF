import tensorflow as tf
import numpy as np
import os
import sys
import random
import subprocess
from redis import Redis

sys.path.append(os.path.realpath(".."))
import json
from redis import Redis
import utils as hlp
from models.feed_forward import *
import math
# Shut up tensorflow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.app.flags.DEFINE_string("config", "config", "Index of task within the job")

FLAGS = tf.app.flags.FLAGS
with open('configs/' + FLAGS.config, 'r') as fp:
    config = json.load(fp)

algo = hlp.agent_from_config(config)
env = hlp.env_from_config(config)
config['environment'] = env
sess = tf.InteractiveSession()
config['n_features'] = env.get_observation_space()
config['n_actions'] = env.get_action_space()
agent = algo(sess, config)
agent.work()