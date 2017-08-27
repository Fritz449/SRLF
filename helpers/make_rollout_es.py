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

for weight in agent.weights:
    phs.append(tf.placeholder(shape=weight.get_shape(), dtype=tf.float32))
    set_op.append(weight.assign(phs[-1]))

seeds = hlp.load_object(variables_server.get("seeds"))
means = hlp.load_object(variables_server.get("means"))
stds = hlp.load_object(variables_server.get("stds"))
sess.run(agent.norm_set_op, feed_dict=dict(zip(agent.norm_phs, [means, stds])))

real_weights = [hlp.load_object(variables_server.get("weight_{}".format(i)))
                for i in range(len(agent.weights))]

for id_task_for_worker in range(agent.n_tasks):
    id_task = id_task_for_worker * agent.n_workers + FLAGS.worker_index
    report_every = math.ceil(agent.n_tasks * agent.n_workers / 16)
    if id_task % report_every == 0:
        print("Rollout # {} of {}, report every {}.".format(id_task, agent.n_tasks * agent.n_workers,
                                                            report_every))
    seed = seeds[id_task]
    np.random.seed(seed)
    noises = []
    for i, weight in enumerate(real_weights):
        noise = np.random.normal(size=weight.shape)
        noises.append(noise)
        real_weights[i] += agent.noise_scale * noise
    agent.set_weights(real_weights)

    sums = np.zeros((1, env.get_observation_space()))
    sumsqrs = np.zeros(sums.shape)

    env.reset()
    while not env.done and env.timestamp < config['max_pathlength']:
        sums += env.features
        sumsqrs += np.square(env.features)

        actions = agent.act(env.features)
        env.step(actions)

    variables_server.set("scores_{}".format(id_task),
                         hlp.dump_object(env.get_total_reward()))
    variables_server.set("eplen_{}".format(id_task), hlp.dump_object(env.timestamp))

    for i, weight in enumerate(real_weights):
        noise = noises[i]
        real_weights[i] -= 2 * agent.noise_scale * noise
    agent.set_weights(real_weights)

    env.reset()
    while not env.done and env.timestamp < config['max_pathlength']:
        sums += env.features
        sumsqrs += np.square(env.features)
        actions = agent.act(env.features)
        env.step(actions)

    variables_server.set("scores_{}".format(-id_task),
                         hlp.dump_object(env.get_total_reward()))
    variables_server.set("eplen_{}".format(-id_task), hlp.dump_object(env.timestamp))
    variables_server.set("sum_{}".format(id_task), hlp.dump_object(sums))
    variables_server.set("sumsqr_{}".format(id_task), hlp.dump_object(sumsqrs))
    for i, weight in enumerate(real_weights):
        noise = noises[i]
        real_weights[i] += agent.noise_scale * noise
    agent.set_weights(real_weights)
