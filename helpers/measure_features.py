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
tf.app.flags.DEFINE_boolean("measuring", False, "Measuring env params")
tf.app.flags.DEFINE_integer("worker_index", 0, "Index of task within the job")


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
variables_server = Redis(port=12000)

phs = []
set_op = []

for id_task_for_worker in range(config['n_pre_tasks']):
    id_task = id_task_for_worker * config['n_workers'] + FLAGS.worker_index
    sums = np.zeros((1, env.get_observation_space()))
    sumsqrs = np.zeros(sums.shape)
    timestamp = 0.
    env.reset()
    while not env.done and timestamp < config['max_pathlength']:
        sums += env.features
        sumsqrs += np.square(env.features)

        actions = agent.act(env.features)
        env.step(actions)
        timestamp += 1

    variables_server.set("sum_{}".format(id_task), hlp.dump_object(sums))
    variables_server.set("sumsqr_{}".format(id_task), hlp.dump_object(sumsqrs))
    variables_server.set("time_{}".format(id_task), hlp.dump_object(timestamp))
