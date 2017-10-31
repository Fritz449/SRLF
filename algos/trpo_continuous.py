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


class TRPOContinuousTrainer(FFContinuous):
    def __init__(self, sess, args):
        FFContinuous.__init__(self, sess, args)
        self.sess = sess
        self.config = args['config']
        self.env = args['environment']
        self.timesteps_per_launch = args['max_pathlength']
        self.n_workers = args['n_workers']
        self.timesteps_per_batch = args['timesteps_batch']
        self.n_pre_tasks = args['n_pre_tasks']
        self.n_tests = args['n_tests']
        self.max_kl = args['max_kl']
        self.ranks = args['ranks']
        self.scale = args['scale']
        self.gamma = args['gamma']
        self.save_every = args.get('save_every', 1)
        self.sums = self.sumsqrs = self.sumtime = 0
        self.timestep = 0
        self.std = args['std']
        self.create_internal()
        self.init_weights()
        self.train_scores = []
        np.set_printoptions(precision=6)

    def create_internal(self):
        self.targets = {
            "advantage": tf.placeholder(dtype=tf.float32, shape=[None]),
            "return": tf.placeholder(dtype=tf.float32, shape=[None]),
            "action": tf.placeholder(dtype=tf.float32, shape=[None, len(self.n_actions)]),
            "old_mean": tf.placeholder(dtype=tf.float32, shape=[None, len(self.n_actions)]),
            "old_std": tf.placeholder(dtype=tf.float32, shape=[None, len(self.n_actions)]),
            "flat_tangent": tf.placeholder(dtype=tf.float32, shape=[None])
        }
        actions = self.targets["action"]
        action_means = self.action_means
        action_stds = self.action_stds
        old_action_means = self.targets["old_mean"]
        old_action_stds = self.targets["old_std"]

        log_p = tf.reduce_sum(- 0.5 * tf.square((actions - action_means) / action_stds) \
                              - 0.5 * tf.log(2 * np.pi) - tf.log(action_stds), axis=1)

        log_old_p = tf.reduce_sum(- 0.5 * tf.square((actions - old_action_means) / old_action_stds) \
                                  - 0.5 * tf.log(2 * np.pi) - tf.log(old_action_stds), axis=1)

        ratio_n = tf.exp(log_p - log_old_p)
        self.loss = -tf.reduce_mean(ratio_n * self.targets["advantage"])

        def KL_gauss(mean1, std1, mean2, std2):
            return - 0.5 * len(self.n_actions) + tf.reduce_mean(tf.reduce_sum(
                tf.log(std2) - tf.log(std1) + (tf.square(std1) + tf.square(mean1 - mean2)) / (2 * tf.square(std2)),
                axis=1))

        self.KL = KL_gauss(old_action_means, old_action_stds, action_means, action_stds)

        self.policy_grad = hlp.flatgrad(self.loss, self.weights)
        fixed_means = tf.stop_gradient(action_means)
        fixed_stds = tf.stop_gradient(action_stds)
        KL_firstfixed = KL_gauss(fixed_means, fixed_stds, action_means, action_stds)
        kl_ff_grads = tf.gradients(KL_firstfixed, self.weights)
        w_shapes = list(map(hlp.var_shape, self.weights))
        start = 0
        tangents = []
        for shape in w_shapes:
            size = np.prod(shape)
            param = tf.reshape(self.targets["flat_tangent"][start:(start + size)], shape)
            tangents.append(param)
            start += size
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(kl_ff_grads, tangents)]
        self.fisher_vector_product = hlp.flatgrad(gvp, self.weights)

        self.get_flat = hlp.GetFlat(self.weights, self.sess)
        self.set_from_flat = hlp.SetFromFlat(self.weights, self.sess)

        value_loss = tf.reduce_mean((self.targets["return"] - self.value) ** 2)

        self.value_train_op = tf.train.AdamOptimizer(0.05).minimize(value_loss, var_list=self.value_weights)

    def save(self, name):
        directory = 'saves/' + name + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        directory += 'iteration_{}'.format(self.timestep) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        for i, tensor in enumerate(tf.global_variables()):
            value = self.sess.run(tensor)
            np.save(directory + 'weight_{}'.format(i), value)

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

            for i, tensor in enumerate(tf.global_variables()):
                arr = np.load(directory + 'weight_{}.npy'.format(i))
                self.sess.run(tensor.assign(arr))

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
        self.sess.run(tf.global_variables_initializer())
        init_weights = [self.sess.run(w) for w in self.weights]
        for i in range(len(init_weights)):
            if self.std == "Param":
                for i in range(len(init_weights))[-len(self.n_actions):]:
                    init_weights[i] /= 10.
            if self.std == "Train":
                for i in range(len(init_weights))[-2*len(self.n_actions)::2]:
                    init_weights[i] /= 10.

        self.set_weights(init_weights)

    def train(self):
        cmd_server = 'redis-server --port 12000'
        p = subprocess.Popen(cmd_server, shell=True, preexec_fn=os.setsid)
        self.variables_server = Redis(port=12000)
        means = "-"
        stds = "-"
        if self.scale:
            if self.timestep == 0:
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
            weights = self.get_weights()
            for i, weight in enumerate(weights):
                self.variables_server.set("weight_" + str(i), hlp.dump_object(weight))
            worker_args = \
                {
                    'config': self.config,
                    'n_workers': self.n_workers
                }
            hlp.launch_workers(worker_args, 'helpers/make_rollout.py')

            paths = []
            for i in range(self.n_workers):
                paths += hlp.load_object(self.variables_server.get("paths_{}".format(i)))

            observations = np.concatenate([path["observations"] for path in paths])
            actions = np.concatenate([path["action_tuples"] for path in paths])
            action_means = []
            action_stds = []
            returns = []
            advantages = []
            for path in paths:
                self.sums += path["sumobs"]
                self.sumsqrs += path["sumsqrobs"]
                self.sumtime += path["rewards"].shape[0]
                action_means += [d[0] for d in path["dist_tuples"]]
                action_stds += [d[1] for d in path["dist_tuples"]]
                returns += hlp.discount(path["rewards"], self.gamma, path["timestamps"]).tolist()
                values = self.sess.run(self.value, feed_dict={self.state_input: path["observations"]})
                values = np.append(values, 0 if path["terminated"] else values[-1])
                deltas = (path["rewards"] + self.gamma * values[1:] - values[:-1])
                advantages += hlp.discount(deltas, self.gamma, path["timestamps"]).tolist()
            returns = np.array(returns)
            advantages = np.array(advantages)
            action_means = np.array(action_means)
            action_stds = np.array(action_stds)

            if self.ranks:
                ranks = np.zeros_like(advantages)
                ranks[np.argsort(advantages)] = np.arange(ranks.shape[0], dtype=np.float32) / (ranks.shape[0] - 1)
                ranks -= 0.5
                advantages = ranks[:]
            else:
                advantages -= np.mean(advantages)
                advantages /= (np.std(advantages, ddof=1) + 0.001)

            feed_dict = {self.state_input: observations,
                         self.targets["return"]: returns,
                         self.targets["advantage"]: advantages,
                         self.targets["old_mean"]: action_means,
                         self.targets["old_std"]: action_stds,
                         self.targets["action"]: actions}

            for i in range(5):
                self.sess.run(self.value_train_op, feed_dict)

            train_rewards = np.array([path["rewards"].sum() for path in paths])
            train_lengths = np.array([len(path["rewards"]) for path in paths])

            thprev = self.get_flat()
            def fisher_vector_product(p):
                feed_dict[self.targets["flat_tangent"]] = p
                return self.sess.run(self.fisher_vector_product, feed_dict) + 0.1 * p

            g = self.sess.run(self.policy_grad, feed_dict)
            stepdir = hlp.conjugate_gradient(fisher_vector_product, -g)

            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / self.max_kl)
            fullstep = stepdir / (lm + 1e-18)
            def loss_kl(th):
                self.set_from_flat(th)
                return self.sess.run([self.loss, self.KL], feed_dict=feed_dict)

            theta = hlp.linesearch(loss_kl, thprev, fullstep, self.max_kl)
            self.set_from_flat(theta)

            lossafter, kloldnew = self.sess.run([self.loss, self.KL], feed_dict=feed_dict)

            print("Time for testing!")

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
            total_rewards = np.array([path["rewards"].sum() for path in paths])
            eplens = np.array([len(path["rewards"]) for path in paths])

            if self.scale:
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
KL between old and new     {kl}
Loss after update          {loss}
Mean of features:          {means}
Std of features:           {stds}
Time for iteration:        {tt}
-------------------------------------------------------------
                """.format(
                means=means,
                stds=stds,
                test_scores=np.mean(total_rewards),
                test_eplengths=np.mean(eplens),
                train_scores=np.mean(train_rewards),
                train_eplengths=np.mean(train_lengths),
                max_test=np.max(total_rewards),
                max_train=np.max(train_rewards),
                kl=kloldnew,
                loss=lossafter,
                tt=time.time() - start_time
            ))
            if self.timestep % self.save_every == 0:
                self.save(self.config[:-5])
            self.timestep += 1
            self.train_scores.append(np.mean(train_rewards))
