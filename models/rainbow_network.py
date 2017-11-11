import tensorflow as tf

import os
import sys

sys.path.append(os.path.realpath(".."))

from helpers.layers import denselayer, noisy_denselayer
from models.base_model import BaseModel
import numpy as np


class RainbowNetwork(BaseModel):
    def __init__(self, sess, args):
        BaseModel.__init__(self, sess)
        self.gamma = args['gamma']
        self.n_hiddens = args['n_hiddens']
        self.n_actions = args['n_actions']
        self.dueling = args['dueling']
        self.noisy = args['noisy_nn']
        self.factorized_noise = args['factorized_noise']
        if len(self.n_actions) > 1:
            print("Unfortunately, Rainbow doesn't support multiple actions.")
            raise Exception
        self.n_actions = self.n_actions[0]
        self.n_atoms = args['n_atoms']

        self.n_features = args['n_features']
        self.nonlinearity = args.get('nonlin', tf.nn.relu)

        self.state_input = tf.placeholder(tf.float32, shape=(None, self.n_features))
        self.next_state_input = tf.placeholder(tf.float32, shape=(None, self.n_features))

        self.create_networks()
        self.sess.run(tf.global_variables_initializer())

        self.set_op = []
        self.target_set_op = []

        for weight, ph in zip(self.weights, self.weights_phs):
            self.set_op.append(weight.assign(ph))
        for weight, ph in zip(self.target_weights, self.target_weights_phs):
            self.target_set_op.append(weight.assign(ph))

    def create_network(self, name, state_input):
        hidden = state_input
        weights = []
        with tf.variable_scope(name):
            for index, n_hidden in enumerate(self.n_hiddens):
                if self.noisy:
                    hidden, layer_weights = noisy_denselayer("hidden_{}".format(index), hidden, n_hidden, self.nonlinearity, self.factorized_noise)
                else:
                    hidden, layer_weights = denselayer("hidden_{}".format(index), hidden, n_hidden, self.nonlinearity)
                weights += layer_weights
            if self.dueling:
                if self.noisy:
                    atom_probs_advs, layer_weights = noisy_denselayer("atom_probs", hidden, self.n_actions * self.n_atoms, factorized=self.factorized_noise)
                else:
                    atom_probs_advs, layer_weights = denselayer("atom_probs", hidden, self.n_actions * self.n_atoms)
                atom_probs_advs = tf.reshape(atom_probs_advs, [-1, self.n_actions, self.n_atoms])
                weights += layer_weights
                if self.noisy:
                    atom_probs_values, layer_weights = noisy_denselayer("atom_probs_values", hidden, self.n_atoms, factorized=self.factorized_noise)
                else:
                    atom_probs_values, layer_weights = denselayer("atom_probs_values", hidden, self.n_atoms)
                atom_probs_values = tf.reshape(atom_probs_values, [-1, 1, self.n_atoms])
                weights += layer_weights
                atom_probs_logs = atom_probs_values + atom_probs_advs - tf.reduce_mean(atom_probs_advs, axis=1, keep_dims=True)
                atom_probs = tf.nn.log_softmax(atom_probs_logs, dim=2)
            else:
                if self.noisy:
                    atom_probs_logs, layer_weights = noisy_denselayer("atom_probs", hidden, self.n_actions * self.n_atoms, factorized=self.factorized_noise)
                else:
                    atom_probs_logs, layer_weights = denselayer("atom_probs", hidden, self.n_actions * self.n_atoms)

                atom_probs = tf.nn.log_softmax(tf.reshape(atom_probs_logs, [-1, self.n_actions, self.n_atoms]), dim=2)

                weights += layer_weights
            weight_phs = [tf.placeholder(tf.float32, shape=w.get_shape()) for w in weights]
        return atom_probs, weights, weight_phs

    def create_networks(self):
        self.mean = tf.get_variable("means", shape=(1, int(self.state_input.get_shape()[1])),
                                    initializer=tf.constant_initializer(0),
                                    trainable=False)
        self.std = tf.get_variable("stds", shape=(1, int(self.state_input.get_shape()[1])),
                                   initializer=tf.constant_initializer(1),
                                   trainable=False)

        mean_ph = tf.placeholder(tf.float32, shape=self.mean.get_shape())
        std_ph = tf.placeholder(tf.float32, shape=self.std.get_shape())
        self.norm_set_op = [self.mean.assign(mean_ph), self.std.assign(std_ph)]
        self.norm_phs = [mean_ph, std_ph]
        self.good_input = tf.clip_by_value((self.state_input - self.mean) / (self.std + 1e-5), -50, 50)
        self.good_next_input = tf.clip_by_value((self.next_state_input - self.mean) / (self.std + 1e-5), -50, 50)

        self.atom_probs, self.weights, self.weights_phs = self.create_network("network", self.good_input)
        self.target_atom_probs, self.target_weights, self.target_weights_phs = self.create_network("target",
                                                                                                   self.good_next_input)

    def act(self, obs, exploration=False):
        if exploration and np.random.randint(100)<1:
            greedy_action = np.random.randint(self.n_actions)
        else:
            atom_probs = self.sess.run(self.atom_probs, feed_dict={self.state_input: obs})
            greedy_action = np.argmax(np.mean(np.exp(atom_probs[0])*np.arange(self.n_atoms), axis=1))
        return greedy_action

    def get_target_weights(self):
        return self.sess.run(self.target_weights)

    def set_target_weights(self, new_weights):
        self.sess.run(self.target_set_op, feed_dict=dict(zip(self.target_weights_phs, new_weights)))
