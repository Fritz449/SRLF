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
        self.gamma = args['gamma']
        self.n_hiddens = args['n_hiddens']
        self.n_actions = args['n_actions']
        self.n_features = args['n_features']
        self.std = args['action_noise']
        self.nonlinearity = args.get('nonlin', tf.nn.relu)

        self.state_input = tf.placeholder(tf.float32, shape=(None, self.n_features))
        self.next_state_input = tf.placeholder(tf.float32, shape=(None, self.n_features))
        self.reward_input = tf.placeholder(tf.float32, shape=(None,))
        self.done_input = tf.placeholder(tf.float32, shape=(None,))
        self.action_input = tf.placeholder(tf.float32, shape=(None, len(self.n_actions)))

        self.create_networks()
        self.sess.run(tf.global_variables_initializer())

        self.set_op = []
        self.value_set_op = []
        self.target_set_op = []
        self.target_value_set_op = []

        for weight, ph in zip(self.weights, self.weights_phs):
            self.set_op.append(weight.assign(ph))
        for weight, ph in zip(self.value_weights, self.value_weights_phs):
            self.value_set_op.append(weight.assign(ph))
        for weight, ph in zip(self.target_weights, self.target_weights_phs):
            self.target_set_op.append(weight.assign(ph))
        for weight, ph in zip(self.target_value_weights, self.target_value_weights_phs):
            self.target_value_set_op.append(weight.assign(ph))

    def create_actor(self, name, state_input):
        hidden = state_input
        weights = []
        with tf.variable_scope(name):
            for index, n_hidden in enumerate(self.n_hiddens):
                hidden, layer_weights = denselayer("hidden_{}".format(index), hidden, n_hidden, self.nonlinearity,
                                                   tf.truncated_normal_initializer())
                weights += layer_weights

            action_means, layer_weights = denselayer("means", hidden, len(self.n_actions), tf.nn.tanh,
                                                     w_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
            weights += layer_weights
            weight_phs = [tf.placeholder(tf.float32, shape=w.get_shape()) for w in weights]
        return action_means, weights, weight_phs

    def create_critic(self, name, state_input, action_input, reuse=False):
        hidden = state_input
        weights = []
        with tf.variable_scope(name, reuse=reuse):
            for index, n_hidden in enumerate(self.n_hiddens):
                if index == 1:
                    hidden = tf.concat([hidden, action_input], axis=1)
                hidden, layer_weights = denselayer("hidden_critic_{}".format(index), hidden, n_hidden,
                                                   self.nonlinearity, tf.truncated_normal_initializer())
                weights += layer_weights

            value, layer_weights = denselayer("value", hidden, 1,
                                              w_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
            value = tf.reshape(value, [-1])
            weights += layer_weights
            weight_phs = [tf.placeholder(tf.float32, shape=w.get_shape()) for w in weights]
        return value, weights, weight_phs

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

        self.good_input = tf.clip_by_value((input - mean) / (std + 1e-5), -20, 20)
        self.good_next_input = tf.clip_by_value((self.next_state_input - mean) / (std + 1e-5), -20, 20)

        self.action_target_means, self.target_weights, self.target_weights_phs = self.create_actor("target_actor",
                                                                                              self.good_next_input)
        self.target_next_value, self.target_value_weights, self.target_value_weights_phs = self.create_critic(
            "target_critic", self.good_next_input, self.action_target_means)
        self.better_value = self.reward_input + self.gamma * (1 - self.done_input) * self.target_next_value

        self.action_means, self.weights, self.weights_phs = self.create_actor("actor", self.good_input)
        self.critic_value, self.value_weights, self.value_weights_phs = self.create_critic("critic", self.good_input,
                                                                                           self.action_input)
        self.value_for_train, self.value_weights, self.value_weights_phs = self.create_critic("critic", self.good_input,
                                                                                              self.action_means,
                                                                                              reuse=True)

    def act(self, obs, exploration=False):
        means = self.sess.run(self.action_means, feed_dict={self.state_input: obs})
        means = means[0] + np.random.normal(size=means[0].shape) * self.std
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
        return self.sess.run(self.target_value_weights)

    def set_target_value_weights(self, new_weights):
        self.sess.run(self.target_value_set_op, feed_dict=dict(zip(self.target_value_weights_phs, new_weights)))
