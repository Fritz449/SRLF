import tensorflow as tf

import os
import sys

sys.path.append(os.path.realpath(".."))

from helpers.layers import denselayer
from models.base_model import BaseModel
import numpy as np


class DDPGNetwork(BaseModel):
    def __init__(self, sess, args):
        BaseModel.__init__(self, sess)
        self.n_hiddens = args['n_hiddens']
        self.n_actions = args['n_actions']
        self.n_features = args['n_features']
        self.nonlinearity = args.get('nonlin', tf.nn.relu)
        self.state_input = tf.placeholder(tf.float32, shape=(None, self.n_features))
        self.action_input = tf.placeholder(tf.float32, shape=(None, len(self.n_actions)))
        self.create_networks()
        self.create_target_networks()
        self.sess.run(tf.global_variables_initializer())

        self.set_op = []
        self.value_set_op = []
        self.target_set_op = []
        self.target_value_set_op = []

        for weight, ph in zip(self.weights, self.weight_phs):
            self.set_op.append(weight.assign(ph))
        for weight, ph in zip(self.value_weights, self.value_weights_phs):
            self.value_set_op.append(weight.assign(ph))
        for weight, ph in zip(self.target_weights, self.target_weights_phs):
            self.target_set_op.append(weight.assign(ph))
        for weight, ph in zip(self.target_value_weights, self.target_value_weights_phs):
            self.target_value_set_op.append(weight.assign(ph))

    def create_networks(self):
        input = self.state_input
        mean = tf.get_variable("means", shape=(1, int(input.get_shape()[1])), initializer=tf.constant_initializer(0),
                               trainable=False)
        std = tf.get_variable("stds", shape=(1, int(input.get_shape()[1])), initializer=tf.constant_initializer(1),
                              trainable=False)

        mean_ph = tf.placeholder(tf.float32, shape=mean.get_shape())
        std_ph = tf.placeholder(tf.float32, shape=std.get_shape())
        self.norm_set_op = [mean.assign(mean_ph), std.assign(std_ph)]
        self.norm_phs = [mean_ph, std_ph]
        hidden = (input - mean) / (std + 1e-5)
        hidden = tf.clip_by_value(hidden, -5, 5)
        self.good_input = hidden

        self.value_weights = []
        self.value_weights_phs = []
        # Actor
        hidden = self.good_input
        for index, n_hidden in enumerate(self.n_hiddens):
            hidden, weights = denselayer("hidden_{}".format(index), hidden, n_hidden, self.nonlinearity)
            self.weights += weights
            self.weight_phs += [tf.placeholder(tf.float32, shape=w.get_shape()) for w in weights]

        self.action_means, weights = denselayer("means", hidden, len(self.n_actions))
        self.weights += weights
        self.weight_phs += [tf.placeholder(tf.float32, shape=w.get_shape()) for w in weights]

        # Critic
        hidden = self.good_input
        for index, n_hidden in enumerate(self.n_hiddens):
            hidden, weights = denselayer("hidden_critic_{}".format(index), hidden, n_hidden, self.nonlinearity)
            self.value_weights += weights
            self.value_weights_phs += [tf.placeholder(tf.float32, shape=w.get_shape()) for w in weights]

        hidden = tf.concat([hidden, self.action_input], axis=1)
        self.value, weights = denselayer("value", hidden, 1)
        self.value = tf.reshape(self.value, [-1])
        self.value_weights += weights
        self.value_weights_phs += [tf.placeholder(tf.float32, shape=w.get_shape()) for w in weights]

    def create_target_networks(self):

        self.target_weights = []
        self.target_weights_phs = []
        self.target_value_weights = []
        self.target_value_weights_phs = []

        # Actor
        hidden = self.good_input
        for index, n_hidden in enumerate(self.n_hiddens):
            hidden, weights = denselayer("target_hidden_{}".format(index), hidden, n_hidden, self.nonlinearity)
            self.target_weights += weights
            self.target_weights_phs += [tf.placeholder(tf.float32, shape=w.get_shape()) for w in weights]

        self.action_target_means, weights = denselayer("target_means", hidden, len(self.n_actions))
        self.target_weights += weights
        self.target_weights_phs += [tf.placeholder(tf.float32, shape=w.get_shape()) for w in weights]

        # Critic
        hidden = self.good_input
        for index, n_hidden in enumerate(self.n_hiddens):
            hidden, weights = denselayer("target_hidden_critic_{}".format(index), hidden, n_hidden, self.nonlinearity)
            self.target_value_weights += weights
            self.target_value_weights_phs += [tf.placeholder(tf.float32, shape=w.get_shape()) for w in weights]

        hidden = tf.concat([hidden, self.action_input], axis=1)
        self.target_value, weights = denselayer("target_value", hidden, 1)
        self.target_value = tf.reshape(self.target_value, [-1])
        self.target_value_weights += weights
        self.target_value_weights_phs += [tf.placeholder(tf.float32, shape=w.get_shape()) for w in weights]

    def act(self, obs, exploration=False):
        means = self.sess.run(self.action_means, feed_dict={self.state_input: obs})
        means = means[0]
        return means

    def get_value_weights(self):
        return self.sess.run(self.value_weights)

    def set_value_weights(self, new_weights):
        self.sess.run(self.value_set_op, feed_dict=dict(zip(self.value_weights_phs, new_weights)))

    def get_target_weights(self):
        return self.sess.run(self.target_weights)

    def set_target_weights(self, new_weights):
        self.sess.run(self.target_set_op, feed_dict=dict(zip(self.target_weights_phs, new_weights)))

    def get_target_value_weights(self):
        return self.sess.run(self.value_weights)

    def set_target_value_weights(self, new_weights):
        self.sess.run(self.target_value_set_op, feed_dict=dict(zip(self.target_value_weights_phs, new_weights)))

