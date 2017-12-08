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
from models.feed_forward import FFDiscrete


class A3CDiscreteTrainer(FFDiscrete):
    def __init__(self, sess, args):
        FFDiscrete.__init__(self, sess, args)
        self.sess = sess
        self.config = args['config']
        self.env = args['environment']
        self.timesteps_per_launch = args['max_pathlength']
        self.n_workers = args['n_workers']
        self.distributed = args['distributed']
        self.n_tests = args['n_tests']
        self.entropy_coef = args['entropy_coef']
        self.learning_rate = args['learning_rate']
        self.n_steps = args['n_steps']
        self.scale = args['scale']
        self.gamma = args['gamma']
        self.save_every = args.get('save_every', 1)
        self.test_every = args.get('test_every', 10)

        self.sums = self.sumsqrs = self.sumtime = 0
        self.timestep = 0
        self.create_internal()
        self.train_scores = []
        self.test_scores = []
        np.set_printoptions(precision=6)

        # Worker parameters:
        self.id_worker = args['id_worker']
        self.test_mode = args['test_mode']

    def create_internal(self):
        self.targets = {
            "advantage": tf.placeholder(dtype=tf.float32, shape=[None]),
            "return": tf.placeholder(dtype=tf.float32, shape=[None]),
        }
        for i in range(len(self.n_actions)):
            self.targets["action_{}".format(i)] = tf.placeholder(dtype=tf.int32, shape=[None])

        N = tf.shape(self.targets["advantage"])[0]
        base = [N] + [1 for _ in range(len(self.n_actions))]
        log_dist = tf.zeros(shape=[N] + self.n_actions)
        p_n = tf.zeros(shape=[N])
        for i, n in enumerate(self.n_actions):
            right_shape = base[:]
            right_shape[i + 1] = n
            actions = self.targets["action_{}".format(i)]
            action_log_dist = tf.reshape(self.action_logprobs[i], [-1])
            p = tf.reshape(tf.gather(action_log_dist, tf.range(0, N) * n + actions), [-1])
            p_n += p
            log_dist += tf.reshape(action_log_dist, right_shape)

        N = tf.cast(N, tf.float32)

        self.loss = -tf.reduce_mean(p_n * self.targets["advantage"])
        self.entropy = tf.reduce_sum(-tf.exp(log_dist) * log_dist) / N

        value_loss = tf.reduce_mean((self.targets["return"] - self.value) ** 2)

        self.loss += -self.entropy_coef * self.entropy + value_loss / 2

        self.weights += self.value_weights
        self.gradients = tf.gradients(self.loss, self.weights)

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

        if self.scale != 'off':
            np.save(directory + 'sums', self.sums)
            np.save(directory + 'sumsquares', self.sumsqrs)
            np.save(directory + 'sumtime', self.sumtime)

        np.save(directory + 'timestep', np.array([self.timestep]))
        np.save(directory + 'train_scores', np.array(self.train_scores))
        np.save(directory + 'test_scores', np.array(self.test_scores))
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

            if self.scale != 'off':
                self.sums = np.load(directory + 'sums.npy')
                self.sumsqrs = np.load(directory + 'sumsquares.npy')
                self.sumtime = np.load(directory + 'sumtime.npy')

            self.timestep = np.load(directory + 'timestep.npy')[0]
            self.train_scores = np.load(directory + 'train_scores.npy').tolist()
            self.test_scores = np.load(directory + 'test_scores.npy').tolist()

            print("Agent successfully loaded from folder {}".format(directory))
        except:
            print("Something is wrong, loading failed")

    def apply_adam_updates(self, variables_server, gradients, learning_rate, epsilon=1e-6):
        update_steps = hlp.load_object(variables_server.get('update_steps')) + 1
        variables_server.set('update_steps', hlp.dump_object(update_steps))
        learning_rate = learning_rate * ((1 - 0.999 ** update_steps) ** 0.5) / (1 - 0.9 ** update_steps)
        for i, gradient in enumerate(gradients):
            momentum = hlp.load_object(variables_server.get('momentum_{}'.format(i)))
            momentum = 0.999 * momentum + (1 - 0.999) * gradient * gradient
            variables_server.set('momentum_{}'.format(i), hlp.dump_object(momentum))
            velocity = hlp.load_object(variables_server.get('velocity_{}'.format(i)))
            velocity = 0.9 * velocity + (1 - 0.9) * gradient
            variables_server.set('velocity_{}'.format(i), hlp.dump_object(velocity))
            weight = hlp.load_object(variables_server.get('weight_{}'.format(i)))
            new_weight = weight - velocity * learning_rate / ((momentum ** 0.5) + epsilon)
            variables_server.set('weight_{}'.format(i), hlp.dump_object(new_weight))
        return update_steps

    def work(self):
        variables_server = Redis(port=12000)
        if self.scale != 'off':
            try:
                means = hlp.load_object(variables_server.get("means"))
                stds = hlp.load_object(variables_server.get("stds"))
                self.sess.run(self.norm_set_op, feed_dict=dict(zip(self.norm_phs, [means, stds])))
            except:
                pass
        try:
            weights = [hlp.load_object(variables_server.get("weight_{}".format(i))) for i in
                       range(len(self.weights))]
            self.set_weights(weights)
        except:
            pass
        env = self.env

        while True:
            observations, action_tuples, rewards, timestamps = [], [], [], []
            for _ in range(self.n_steps):
                observations.append(env.features[0])
                timestamps.append(env.timestamp)

                actions = self.act(env.features)
                env.step(actions)

                action_tuples.append(actions)
                rewards.append(env.reward)
                if env.done or env.timestamp > self.timesteps_per_launch:
                    variables_server.lpush('results', hlp.dump_object(env.get_total_reward()))
                    print("Episode reward: {}".format(env.get_total_reward()), "Length: {}".format(env.timestamp))
                    break
            timestamps.append(env.timestamp)

            observations_batch = np.array(observations)
            actions_batch = np.array(action_tuples)

            feed_dict = {self.state_input: observations_batch}
            for i in range(len(self.n_actions)):
                feed_dict[self.targets["action_{}".format(i)]] = actions_batch[:, i]

            if env.done or env.timestamp > self.timesteps_per_launch:
                rewards.append(0)
                env.reset()
            else:
                obs = observations[-1]
                rewards.append(self.sess.run(self.value, feed_dict={self.state_input: obs.reshape((1,) + obs.shape)}))
            returns_batch = hlp.discount(np.array(rewards), self.gamma, np.array(timestamps))[:-1]
            values = self.sess.run(self.value, feed_dict)
            feed_dict[self.targets["advantage"]] = returns_batch - values
            feed_dict[self.targets["return"]] = returns_batch
            gradients = self.sess.run(self.gradients, feed_dict)
            self.apply_adam_updates(variables_server, gradients, self.learning_rate)
            weights = [hlp.load_object(variables_server.get("weight_{}".format(i))) for i in
                       range(len(self.weights))]
            self.set_weights(weights)

    def make_rollout(self):
        variables_server = Redis(port=12000)
        if self.scale != 'off':
            try:
                means = hlp.load_object(variables_server.get("means"))
                stds = hlp.load_object(variables_server.get("stds"))
                self.sess.run(self.norm_set_op, feed_dict=dict(zip(self.norm_phs, [means, stds])))
            except:
                pass
        try:
            weights = [hlp.load_object(variables_server.get("weight_{}".format(i))) for i in
                       range(len(self.weights))]
            self.set_weights(weights)
        except:
            pass
        env = self.env
        n_tasks = self.n_tests

        timestep = 0
        i_task = 0

        paths = []
        while i_task < n_tasks:
            path = {}
            observations, action_tuples, rewards, dist_tuples, timestamps = [], [], [], [], []
            sums = np.zeros((1, env.get_observation_space()))
            sumsqrs = np.zeros(sums.shape)

            env.reset()
            while not env.done and env.timestamp < self.timesteps_per_launch:
                sums += env.features
                sumsqrs += np.square(env.features)
                observations.append(env.features[0])
                timestamps.append(env.timestamp)

                if not self.test_mode:
                    actions, dist_tuple = self.act(env.features, return_dists=True)
                    dist_tuples.append(dist_tuple)
                else:
                    actions = self.act(env.features, exploration=False)
                env.step(actions)
                timestep += 1

                action_tuples.append(actions)
                rewards.append(env.reward)

            path["observations"] = np.array(observations)
            path["action_tuples"] = np.array(action_tuples)
            path["rewards"] = np.array(rewards)
            if not self.test_mode:
                path["dist_tuples"] = np.array(dist_tuples)
            path["timestamps"] = np.array(timestamps)
            path["sumobs"] = sums
            path["sumsqrobs"] = sumsqrs
            path["terminated"] = env.done
            path["total"] = env.get_total_reward()
            paths.append(path)
            i_task += 1

        if self.distributed:
            variables_server.set("paths_{}".format(self.id_worker), hlp.dump_object(paths))
        else:
            self.paths = paths

    def train(self):
        cmd_server = 'redis-server --port 12000'
        p = subprocess.Popen(cmd_server, shell=True, preexec_fn=os.setsid)
        self.variables_server = Redis(port=12000)
        means = "-"
        stds = "-"
        if self.scale != 'off':
            if self.timestep == 0:
                print("Time to measure features!")
                if self.distributed:
                    worker_args = \
                        {
                            'config': self.config,
                            'test_mode': False,
                        }
                    hlp.launch_workers(worker_args, self.n_workers)
                    paths = []
                    for i in range(self.n_workers):
                        paths += hlp.load_object(self.variables_server.get("paths_{}".format(i)))
                else:
                    self.test_mode = False
                    self.make_rollout()
                    paths = self.paths

                for path in paths:
                    self.sums += path["sumobs"]
                    self.sumsqrs += path["sumsqrobs"]
                    self.sumtime += path["observations"].shape[0]

            stds = np.sqrt((self.sumsqrs - np.square(self.sums) / self.sumtime) / (self.sumtime - 1))
            means = self.sums / self.sumtime
            print("Init means: {}".format(means))
            print("Init stds: {}".format(stds))
            self.variables_server.set("means", hlp.dump_object(means))
            self.variables_server.set("stds", hlp.dump_object(stds))
            self.sess.run(self.norm_set_op, feed_dict=dict(zip(self.norm_phs, [means, stds])))

        weights = self.get_weights()
        for i, weight in enumerate(weights):
            self.variables_server.set("weight_" + str(i), hlp.dump_object(weight))
            self.variables_server.set('momentum_{}'.format(i), hlp.dump_object(np.zeros(weight.shape)))
            self.variables_server.set('velocity_{}'.format(i), hlp.dump_object(np.zeros(weight.shape)))
        self.variables_server.set('update_steps', hlp.dump_object(0))

        worker_args = \
            {
                'config': self.config,
                'test_mode': False,
            }
        hlp.launch_workers(worker_args, self.n_workers, command='work', wait=False)

        while True:
            time.sleep(self.test_every)
            print("Time for testing!")
            if self.distributed:
                worker_args = \
                    {
                        'config': self.config,
                        'test_mode': True,
                    }
                hlp.launch_workers(worker_args, self.n_workers)
                paths = []
                for i in range(self.n_workers):
                    paths += hlp.load_object(self.variables_server.get("paths_{}".format(i)))
            else:
                self.test_mode = True
                self.make_rollout()
                paths = self.paths

            total_rewards = np.array([path["total"] for path in paths])
            eplens = np.array([len(path["rewards"]) for path in paths])

            print("""
-------------------------------------------------------------
Mean test score:           {test_scores}
Mean test episode length:  {test_eplengths}
Max test score:            {max_test}
Number of train episodes:  {number}
Mean of features:          {means}
Std of features:           {stds}
-------------------------------------------------------------
                """.format(
                means=means,
                stds=stds,
                test_scores=np.mean(total_rewards),
                test_eplengths=np.mean(eplens),
                max_test=np.max(total_rewards),
                number=self.variables_server.llen('results')
            ))
            self.timestep += 1
            self.train_scores = [hlp.load_object(res) for res in self.variables_server.lrange('results', 0, -1)][::-1]

            self.test_scores.append(np.mean(total_rewards))
            if self.timestep % self.save_every == 0:
                self.save(self.config[:-5])
