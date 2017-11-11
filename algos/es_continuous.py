import tensorflow as tf
import numpy as np
import os
import sys
import random
import subprocess
from redis import Redis
import time

sys.path.append(os.path.realpath(".."))

import helpers.utils as hlp
from models.feed_forward import FFContinuous


class EvolutionStrategiesTrainer(FFContinuous):
    def __init__(self, sess, args):
        FFContinuous.__init__(self, sess, args)
        self.sess = sess
        self.config = args['config']
        self.env = args['environment']
        self.timesteps_per_launch = args['max_pathlength']
        self.noise_scale = args['noise_scale']
        self.n_workers = args['n_workers']
        self.n_tasks_all = args['n_tasks']
        self.n_tasks_all -= self.n_tasks_all % self.n_workers
        self.n_tasks = self.n_tasks_all // self.n_workers
        self.n_pre_tasks = args['n_pre_tasks']
        self.n_tests = args['n_tests']
        self.learning_rate = args['learning_rate']
        self.adam = args['momentum']
        self.ranks = args['ranks']
        self.scale = args['scale']
        self.l1_reg = args.get('l1_reg', 0.005)
        self.save_every = args.get('save_every', 1)
        self.sums = self.sumsqrs = self.sumtime = 0
        self.timestep = 0
        self.velocity = []
        self.momentum = []
        self.train_scores = []
        self.std = args['std']
        for w in self.weights:
            self.velocity.append(np.zeros(shape=w.get_shape()))
            self.momentum.append(np.zeros(shape=w.get_shape()))
        self.init_weights()
        np.set_printoptions(precision=4)

    def save(self, name):
        directory = 'saves/' + name + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        directory += 'iteration_{}'.format(self.timestep) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        for i, w in enumerate(self.weights):
            np.save(directory + 'weight_{}'.format(i), self.sess.run(w))
            np.save(directory + 'velocity_{}'.format(i), self.velocity[i])
            np.save(directory + 'momentum_{}'.format(i), self.momentum[i])

        if self.scale:
            np.save(directory + 'sums', self.sums)
            np.save(directory + 'sumsquares', self.sumsqrs)
            np.save(directory + 'sumtime', self.sumtime)

        np.save(directory + 'timestep', np.array([self.timestep]))
        np.save(directory + 'train_scores', np.array(self.train_scores))
        print("Agent successfully saved in folder {}".format(directory))

    def load(self, name, iteration=None):
        try:
            directory = 'saves/' + name + '/'
            if not os.path.exists(directory):
                print('That directory does not exist!')
                raise Exception
            if iteration is None:
                iteration = np.max([int(x[10:]) for x in [dir for dir in os.walk(directory)][0][1]])
            directory += 'iteration_{}'.format(iteration) + '/'
            weights = [np.zeros(shape=w.get_shape()) for w in self.weights]
            for i in range(len(self.weights)):
                weights[i] = np.load(directory + 'weight_{}.npy'.format(i))
                self.momentum[i] = np.load(directory + 'momentum_{}.npy'.format(i))
                self.velocity[i] = np.load(directory + 'velocity_{}.npy'.format(i))
            self.set_weights(weights)

            if self.scale:
                self.sums = np.load(directory + 'sums.npy')
                self.sumsqrs = np.load(directory + 'sumsquares.npy')
                self.sumtime = np.load(directory + 'sumtime.npy')

            self.timestep = np.load(directory + 'timestep.npy')[0]
            self.train_scores = np.load(directory + 'train_scores.npy').tolist()
            print("Agent successfully loaded from folder {}".format(directory))
        except:
            print("Something is wrong, loading failed")

    def init_weights(self):
        init_weights = [np.zeros(w.get_shape()) for w in self.weights]
        for i in range(len(init_weights))[0::2]:
            init_weights[i] = np.random.normal(size=init_weights[i].shape)
            norms = np.sqrt(np.sum(np.square(init_weights[i]), axis=0, keepdims=True))
            init_weights[i] = init_weights[i] / norms
            if self.std == "Train" and i == len(init_weights) - 4:
                init_weights[i] /= 100.
            if i == len(init_weights) - 2:
                init_weights[i] /= 100.
        if len(init_weights) % 2 == 1:
            init_weights[-1] = self.sess.run(self.weights[-1])
        self.set_weights(init_weights)

    def apply_adam_updates(self, gradients, epsilon=1e-8, beta_1=0.9, beta_2=0.999):
        self.timestep += 1
        learning_rate = self.learning_rate * ((1 - beta_2 ** self.timestep) ** 0.5) / (1 - beta_1 ** self.timestep)
        weights = self.get_weights()
        for i, gradient in enumerate(gradients):
            momentum = self.momentum[i]
            self.momentum[i] = beta_1 * momentum + (1 - beta_1) * gradient
            velocity = self.velocity[i]
            self.velocity[i] = beta_2 * velocity + (1 - beta_2) * gradient * gradient
            weights[i] += self.momentum[i] * learning_rate / ((self.velocity[i] ** 0.5) + epsilon)
        self.set_weights(weights)

    def train(self):
        cmd_server = 'redis-server --port 12000'
        p = subprocess.Popen(cmd_server, shell=True, preexec_fn=os.setsid)
        self.variables_server = Redis(port=12000)
        means = "-"
        stds = "-"
        if self.scale:
            if self.timestep==0:
                print("Time to measure features!")
                worker_args = \
                    {
                        'config': self.config,
                        'n_workers': self.n_workers
                    }
                hlp.launch_workers(worker_args, 'helpers/measure_features.py')
                for i in range(self.n_workers * self.n_pre_tasks):
                    self.sums += hlp.load_object(self.variables_server.get("sum_" + str(i)))
                    self.sumsqrs += hlp.load_object(self.variables_server.get("sumsqr_" + str(i)))
                    self.sumtime += hlp.load_object(self.variables_server.get("time_" + str(i)))
            stds = np.sqrt((self.sumsqrs - np.square(self.sums) / self.sumtime) / (self.sumtime - 1))
            means = self.sums / self.sumtime
            print("Init means: {}".format(means))
            print("Init stds: {}".format(stds))
            self.variables_server.set("means", hlp.dump_object(means))
            self.variables_server.set("stds", hlp.dump_object(stds))
            self.sess.run(self.norm_set_op, feed_dict=dict(zip(self.norm_phs, [means, stds])))

        while True:
            print("Iteration {}".format(self.timestep))
            start_time = time.time()
            weight_noises = []
            random.seed()
            seed_for_random = random.randint(0, np.iinfo(np.int32).max)
            np.random.seed(seed_for_random)
            seeds = np.random.randint(-np.iinfo(np.int32).min + np.iinfo(np.int32).max, size=self.n_tasks_all)
            self.variables_server.set("seeds", hlp.dump_object(seeds))

            weights = self.get_weights()
            for i, weight in enumerate(weights):
                self.variables_server.set("weight_" + str(i), hlp.dump_object(weight))
                weight_noises.append(np.empty((self.n_tasks_all,) + weight.shape))

            for index in range(self.n_tasks_all):
                np.random.seed(seeds[index])
                for i, weight in enumerate(weights):
                    weight_noises[i][index] = np.random.normal(size=weight.shape)

            worker_args = \
                {
                    'config': self.config,
                    'n_workers': self.n_workers
                }
            hlp.launch_workers(worker_args, 'helpers/make_rollout_es.py')

            scores = []
            train_lengths = []
            for i in range(self.n_tasks_all):
                scores.append(hlp.load_object(self.variables_server.get("scores_" + str(i))))
                train_lengths.append(hlp.load_object(self.variables_server.get("eplen_" + str(i))))
                scores.append(hlp.load_object(self.variables_server.get("scores_" + str(-i))))
                train_lengths.append(hlp.load_object(self.variables_server.get("eplen_" + str(-i))))

            scores = np.array(scores)
            train_mean_score = np.mean(scores)
            ranks = np.zeros(shape=scores.shape)
            if self.ranks:
                ranks[np.argsort(scores)] = np.arange(ranks.shape[0], dtype=np.float32) / (
                    ranks.shape[0] - 1)
                ranks -= 0.5
            else:
                ranks = scores[:]
                ranks -= train_mean_score
                ranks /= (np.std(ranks, ddof=1) + 0.001)

            gradients = [np.zeros(w.get_shape()) for w in self.weights]
            for i, weight in enumerate(weights):
                for index in 2 * np.arange(seeds.shape[0]):
                    gradients[i] += weight_noises[i][index // 2] * (ranks[index] - ranks[index + 1]) / self.n_tasks_all
                gradients[i] -= self.l1_reg * weights[i]

            if self.adam:
                self.apply_adam_updates(gradients)
            else:
                for i, weight in enumerate(weights):
                    weights[i] += self.learning_rate * gradients[i]
                self.sess.run(self.set_op, feed_dict=dict(zip(self.weight_phs, weights)))

            print("Time to testing!")
            weights = self.get_weights()
            for i, weight in enumerate(weights):
                self.variables_server.set("weight_" + str(i), hlp.dump_object(weight))
            worker_args = \
                {
                    'config': self.config,
                    'n_workers': self.n_workers,
                    'n_tasks': self.n_tests // self.n_workers,
                    'test': True
                }
            hlp.launch_workers(worker_args, 'helpers/make_rollout.py')
            paths = []
            for i in range(self.n_workers):
                paths += hlp.load_object(self.variables_server.get("paths_{}".format(i)))
            total_rewards = np.array([path["total"] for path in paths])
            eplens = np.array([len(path["rewards"]) for path in paths])

            if self.scale:
                for i in range(self.n_tasks_all):
                    self.sums += hlp.load_object(self.variables_server.get("sum_{}".format(i)))
                    self.sumsqrs += hlp.load_object(self.variables_server.get("sumsqr_{}".format(i)))
                self.sumtime += np.sum(train_lengths)
                stds = np.sqrt((self.sumsqrs - np.square(self.sums) / self.sumtime) / (self.sumtime - 1))
                means = self.sums / self.sumtime
                self.variables_server.set("means", hlp.dump_object(means))
                self.variables_server.set("stds", hlp.dump_object(stds))
                self.sess.run(self.norm_set_op, feed_dict=dict(zip(self.norm_phs, [means, stds])))

            print("""
-------------------------------------------------------------
Mean test score:           {test_scores}
Mean train score:          {train_scores}
Mean test episode length:  {test_eplengths}
Mean train episode length: {train_eplengths}
Max test score:            {max_test}
Max train score:           {max_train}
Mean of features:          {means}
Std of features:           {stds}
Time for iteration:        {tt}
-------------------------------------------------------------
                """.format(
                means=means,
                stds=stds,
                test_scores=np.mean(total_rewards),
                test_eplengths=np.mean(eplens),
                train_scores=train_mean_score,
                train_eplengths=np.mean(train_lengths),
                max_test=np.max(total_rewards),
                max_train=np.max(scores),
                tt=time.time() - start_time
            ))
            self.train_scores.append(train_mean_score)
            if self.timestep % self.save_every == 0:
                self.save(self.config[:-5])
