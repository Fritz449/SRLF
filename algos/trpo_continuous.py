import tensorflow as tf
import numpy as np
import os
import sys
import random
import subprocess
from redis import Redis

sys.path.append(os.path.realpath(".."))

import helpers.utils as hlp
from models.feed_forward import FFContinuous

class ESContinuous(FFContinuous):
    def __init__(self, sess, args):
        FFContinuous.__init__(self, sess, args)
        self.sess = sess
        self.config = args['config']
        self.env = args['environment']
        self.timesteps_per_launch = args['max_pathlength']
        self.n_workers = args['n_workers']
        self.n_tasks_all = args['n_tasks']
        self.n_tasks_all -= self.n_tasks_all % self.n_workers
        self.n_tasks = self.n_tasks_all // self.n_workers
        self.n_pre_tasks = args['n_pre_tasks']
        self.n_tests = args['n_tests']
        self.learning_rate = args['learning_rate']
        self.ranks = args['ranks']
        self.scale = args['scale']
        self.timestamp = 0
        self.velocity = []
        self.momentum = []
        self.std = args['std']

    def act(self, obs, exploration=True):
        means, stds = self.sess.run([self.action_means, self.action_stds], feed_dict={self.state_input: obs})
        if not exploration:
            return means
        actions = np.zeros(shape=means.shape)
        for i in range(actions.shape[0]):
            actions[i] = np.random.normal(means[i], stds[i])
        return actions

