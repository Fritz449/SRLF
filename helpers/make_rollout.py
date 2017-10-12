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
tf.app.flags.DEFINE_integer("worker_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("n_tasks", 100000, "Index of task within the job")
tf.app.flags.DEFINE_boolean("test", False, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS
n_tasks = FLAGS.n_tasks
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
test = FLAGS.test
phs = []
set_op = []

for weight in agent.weights:
    phs.append(tf.placeholder(shape=weight.get_shape(), dtype=tf.float32))
    set_op.append(weight.assign(phs[-1]))

if agent.scale:
    means = hlp.load_object(variables_server.get("means"))
    stds = hlp.load_object(variables_server.get("stds"))
    sess.run(agent.norm_set_op, feed_dict=dict(zip(agent.norm_phs, [means, stds])))

weights = [hlp.load_object(variables_server.get("weight_{}".format(i))) for i in range(len(agent.weights))]
agent.set_weights(weights)
paths = []
timesteps_per_worker = agent.timesteps_per_batch//agent.n_workers
timestep = 0
i_task = 0
while timestep < timesteps_per_worker and i_task < n_tasks:
    path = {}
    observations, action_tuples, rewards, dist_tuples, timestamps = [], [], [], [], []
    sums = np.zeros((1, env.get_observation_space()))
    sumsqrs = np.zeros(sums.shape)

    env.reset()
    while not env.done and env.timestamp < config['max_pathlength']:
        sums += env.features
        sumsqrs += np.square(env.features)
        observations.append(env.features[0])
        timestamps.append(env.timestamp)

        if not test:
            actions, dist_tuple = agent.act(env.features, return_dists=True)
            dist_tuples.append(dist_tuple)
        else:
            actions = agent.act(env.features, exploration=False)
        env.step(actions)
        timestep += 1

        action_tuples.append(actions)
        rewards.append(env.reward)

    path["observations"] = np.array(observations)
    path["action_tuples"] = np.array(action_tuples)
    path["rewards"] = np.array(rewards)
    if not test:
        path["dist_tuples"] = np.array(dist_tuples)
    path["timestamps"] = np.array(timestamps)
    path["sumobs"] = sums
    path["sumsqrobs"] = sumsqrs
    path["terminated"] = env.done
    paths.append(path)
    i_task+=1
variables_server.set("paths_{}".format(FLAGS.worker_index), hlp.dump_object(paths))