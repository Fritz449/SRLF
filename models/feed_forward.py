import tensorflow as tf

import os
import sys

sys.path.append(os.path.realpath(".."))

from helpers.layers import denselayer
from models.base_model import BaseModel
import numpy as np


class FeedForward(BaseModel):
    def __init__(self, sess, args):
        BaseModel.__init__(self, sess)
        self.n_hiddens = args['n_hiddens']
        self.n_features = args['n_features']
        self.critic = args.get('critic')
        self.nonlinearity = args.get('nonlin', tf.nn.tanh)
        self.state_input = tf.placeholder(tf.float32, shape=(None, self.n_features))
        self.value_weights = []
        self.value_weights_phs = []
        self.create_network()

    def create_network(self):
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
        self.hidden = hidden

        for index, n_hidden in enumerate(self.n_hiddens):
            hidden, weights = denselayer("hidden_{}".format(index), hidden, n_hidden, self.nonlinearity)
            self.weights += weights
            self.weight_phs += [tf.placeholder(tf.float32, shape=w.get_shape()) for w in weights]
            self.hidden = hidden

        if self.critic:
            hidden = self.state_input
            for index, n_hidden in enumerate(self.n_hiddens):
                hidden, weights = denselayer("hidden_critic_{}".format(index), hidden, n_hidden, self.nonlinearity)
                self.value_weights += weights
                self.value_weights_phs += [tf.placeholder(tf.float32, shape=w.get_shape()) for w in weights]
            self.value, weights = denselayer("value", hidden, 1)
            self.value = tf.reshape(self.value, [-1])
            self.value_weights += weights
            self.value_weights_phs += [tf.placeholder(tf.float32, shape=w.get_shape()) for w in weights]
            for weight, ph in zip(self.value_weights, self.value_weights_phs):
                self.value_set_op.append(weight.assign(ph))
        else:
            self.value = 0.


class FFDiscrete(FeedForward):
    def __init__(self, sess, args):
        FeedForward.__init__(self, sess, args)
        self.n_actions = args['n_actions']
        self.create_output()
        for weight, ph in zip(self.weights, self.weight_phs):
            self.set_op.append(weight.assign(ph))

    def create_output(self):
        self.action_probs = []
        self.action_logprobs = []
        for index, n in enumerate(self.n_actions):
            log_probs, weights = denselayer("lob_probs_{}".format(index), self.hidden, n, tf.nn.log_softmax)
            self.action_logprobs.append(log_probs)
            self.weights += weights
            self.weight_phs += [tf.placeholder(tf.float32, shape=w.get_shape()) for w in weights]
            self.action_probs.append(tf.exp(self.action_logprobs[index]))

        self.sess.run(tf.global_variables_initializer())

    def act(self, obs, exploration=True, return_dists=False):
        log_probs = self.sess.run(self.action_logprobs, feed_dict={self.state_input: obs})
        actions = np.zeros(shape=(len(log_probs),), dtype=np.int32)
        for i in range(len(log_probs, )):
            if not exploration:
                actions[i] = np.argmax(log_probs[i][0])
                continue
            actions[i] = np.random.choice(np.arange(self.n_actions[i], dtype=np.int32), p=np.exp(log_probs[i][0]))
        if return_dists:
            return actions, log_probs
        return actions


class FFContinuous(FeedForward):
    def __init__(self, sess, args):
        FeedForward.__init__(self, sess, args)
        self.n_actions = args['n_actions']
        self.std = args.get('std', "Const")
        self.init_log_std = args.get('init_log_std', 0)
        self.create_output()
        for weight, ph in zip(self.weights, self.weight_phs):
            self.set_op.append(weight.assign(ph))

    def create_output(self):
        self.action_means, weights = denselayer("means", self.hidden, len(self.n_actions))
        self.weights += weights
        self.weight_phs += [tf.placeholder(tf.float32, shape=w.get_shape()) for w in weights]

        if self.std == "Const":
            self.action_log_stds = tf.get_variable("std", shape=(1, len(self.n_actions),),
                                                   initializer=tf.constant_initializer(self.init_log_std),
                                                   trainable=False)
            self.action_stds = tf.exp(self.action_log_stds)

        elif self.std == "Param":
            self.action_log_stds = tf.get_variable("std", shape=(1, len(self.n_actions)),
                                                   initializer=tf.constant_initializer(self.init_log_std))
            self.action_stds = tf.exp(self.action_log_stds)
            self.weights.append(self.action_log_stds)
            self.weight_phs.append(tf.placeholder(tf.float32, shape=(1, len(self.n_actions))))

        elif self.std == "Train":
            self.action_stds, weights = denselayer("stds", self.hidden, len(self.n_actions), tf.exp)
            self.weights += weights
            self.weight_phs += [tf.placeholder(tf.float32, shape=w.get_shape()) for w in weights]
        else:
            raise Exception

        self.sess.run(tf.global_variables_initializer())

    def act(self, obs, exploration=True, return_dists=False):
        means, stds = self.sess.run([self.action_means, self.action_stds], feed_dict={self.state_input: obs})
        means = means[0]
        stds = stds[0]
        if not exploration:
            return means
        actions = np.zeros(shape=means.shape)
        for i in range(actions.shape[0]):
            actions[i] = np.random.normal(means[i], stds[i])
        if return_dists:
            return actions, [means, stds]
        return actions
