import tensorflow as tf

import os
import sys

sys.path.append(os.path.realpath(".."))
import helpers.utils as hlp
import subprocess
from redis import Redis
import time

from helpers.layers import denselayer
from models.ddpg_network import DDPGNetwork
import numpy as np


class DDPGTrainer(DDPGNetwork):
    def __init__(self, sess, args):
        DDPGNetwork.__init__(self, sess, args)
        self.sess = sess
        self.config = args['config']
        self.env = args['environment']
        self.learning_rate = args['learning_rate']
        self.timesteps_per_launch = args['max_pathlength']
        self.n_workers = args['n_workers']
        self.timesteps_per_batch = args['timesteps_batch']
        self.n_pre_tasks = args['n_pre_tasks']
        self.n_tests = args['n_tests']
        self.ranks = args['ranks']
        self.scale = args['scale']
        self.gamma = args['gamma']
        self.xp_size = args['xp_size']
        self.save_every = args.get('save_every', 1)
        self.batch_size = args['batch_size']
        self.sums = self.sumsqrs = self.sumtime = 0
        self.timestep = 0
        self.create_internal()
        self.sess.run(tf.global_variables_initializer())
        self.train_scores = []
        np.set_printoptions(precision=6)

    def create_internal(self):
        self.targets = {
            "value": tf.placeholder(dtype=tf.float32, shape=[self.batch_size]),
        }
        self.value_loss = tf.reduce_mean((self.targets["value"] - self.value) ** 2)
        l2_reg = 0
        for i in range(len(self.value_weights)):
            l2_reg += 0.01 * tf.reduce_sum(tf.square(self.value_weights[i]))
        self.value_train_op = tf.train.AdamOptimizer(0.001).minimize((self.value_loss+l2_reg), var_list=self.value_weights)
        self.gr_q_a = tf.gradients(self.value, self.action_input)[0]
        gradients = tf.gradients(self.action_means, self.weights,
                                 grad_ys=self.gr_q_a)
        for i in range(len(gradients)):
            gradients[i] = -1 * gradients[i]
        self.train_actor_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(gradients, self.weights))

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

    def load_weights_from_redis(self):
        weights = [hlp.load_object(self.variables_server.get("weight_{}".format(i))) for i in range(len(self.weights))]
        self.set_weights(weights)

    def work(self):
        self.variables_server = Redis(port=12000)
        self.load_weights_from_redis()
        env = self.env

        if self.scale:
            means = hlp.load_object(self.variables_server.get("means"))
            stds = hlp.load_object(self.variables_server.get("stds"))
            self.sess.run(self.norm_set_op, feed_dict=dict(zip(self.norm_phs, [means, stds])))
        st = time.time()
        while True:
            self.last_state = env.reset()
            while not env.done and env.timestamp < self.timesteps_per_launch:
                actions = self.act(env.features)
                env.step(actions)
                transition = hlp.dump_object([self.last_state, env.reward, actions, env.features, env.done])
                self.variables_server.lpush('transitions', transition)
                self.last_state = env.features
                if time.time() - st > 10:
                    st = time.time()
                    self.load_weights_from_redis()
            self.load_weights_from_redis()
            if self.variables_server.llen('transitions') > self.xp_size:
                self.variables_server.ltrim('transitions', 1, self.xp_size + 1)

    def update_target_weights(self, alpha=0.001):
        value_weights = self.get_value_weights()
        new_weights = self.get_target_value_weights()
        for i in range(len(value_weights)):
            new_weights[i] = new_weights[i] * (1 - alpha) + alpha * value_weights[i]
        self.set_target_value_weights(new_weights)

        weights = self.get_weights()
        new_weights = self.get_target_weights()
        for i in range(len(weights)):
            new_weights[i] = new_weights[i] * (1 - alpha) + alpha * weights[i]
        self.set_target_weights(new_weights)

    def train(self):
        cmd_server = 'redis-server --port 12000'
        p = subprocess.Popen(cmd_server, shell=True, preexec_fn=os.setsid)
        self.variables_server = Redis(port=12000)
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
        print("Let's go!")
        weights = self.get_weights()
        for i, weight in enumerate(weights):
            self.variables_server.set("weight_" + str(i), hlp.dump_object(weight))
        worker_args = \
            {
                'config': self.config,
                'n_workers': self.n_workers
            }
        self.set_target_value_weights(self.get_value_weights())
        self.set_target_weights(self.get_weights())

        hlp.launch_workers(worker_args, 'helpers/ddpg_worker.py', wait=False)

        time.sleep(5)
        iteration = 0
        while True:
            start_time = time.time()
            obs_batch = []
            next_obs_batch = []
            done_batch = []
            reward_batch = []
            actions_batch = []
            xp_replay = np.array(self.variables_server.lrange('transitions', 0, self.xp_size))
            idxs = np.random.randint(len(xp_replay), size=self.batch_size)
            transitions_batch = xp_replay[idxs]
            transitions = []
            for i in range(self.batch_size):
                transitions.append(hlp.load_object(transitions_batch[i]))
            for transition in transitions:
                obs_batch.append(transition[0])
                reward_batch.append(transition[1])
                actions_batch.append(transition[2])
                next_obs_batch.append(transition[3])
                done_batch.append(transition[4])
            # print(obs_batch)
            # print (next_obs_batch)
            # print (actions_batch)
            # print (reward_batch)
            # print (done_batch)
            # while time.time()-start_time < 1:
            #     time.sleep(0.001)
            feed_dict = {self.state_input: np.concatenate(next_obs_batch, axis=0)}
            old_means = self.sess.run(self.action_target_means, feed_dict)
            #print (old_means)
            #print(self.sess.run(self.action_means, feed_dict))
            feed_dict[self.action_input] = old_means
            next_state_prediction = self.sess.run(self.target_value, feed_dict)
            done_batch = np.array(done_batch)
            #print(next_state_prediction)

            better_prediction = np.array(reward_batch) + self.gamma * (1 - done_batch) * next_state_prediction
            #print(better_prediction)
            #1/0
            actions = self.sess.run(self.action_means, feed_dict)
            feed_dict = {self.state_input: np.concatenate(obs_batch, axis=0), self.targets['value']: better_prediction,
                         self.action_input: actions}
            print(self.sess.run([self.value_loss, self.value_train_op], feed_dict)[0])

            feed_dict = {self.state_input: np.concatenate(obs_batch, axis=0), self.action_input: actions}
            self.sess.run(self.train_actor_op, feed_dict)

            self.update_target_weights()
            iteration += 1
            if iteration % 100 == 0:
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
                print("""
-------------------------------------------------------------
Mean test score:           {test_scores}
Mean test episode length:  {test_eplengths}
Max test score:            {max_test}
Time for iteration:        {tt}
-------------------------------------------------------------
                                """.format(
                    test_scores=np.mean(total_rewards),
                    test_eplengths=np.mean(eplens),
                    max_test=np.max(total_rewards),
                    tt=time.time() - start_time
                ))
