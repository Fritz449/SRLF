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
        self.l_rate = args['learning_rate']
        self.timesteps_per_launch = args['max_pathlength']
        self.n_workers = args['n_workers']
        self.l_rate_critic = args['learning_rate_critic']
        self.n_pre_tasks = args['n_pre_tasks']
        self.n_tests = args['n_tests']
        self.scale = args['scale']
        self.gamma = args['gamma']
        self.tau = args['tau']
        self.action_noise = args['action_noise']
        self.test_every = args['test_every']
        self.random_steps = args['random_steps']
        self.xp_size = args['xp_size']
        self.save_every = args.get('save_every', 1)
        self.clip_error = args.get('clip_error', 10.)
        self.batch_size = args['batch_size']
        self.sums = self.sumsqrs = self.sumtime = 0
        self.timestep = 0
        self.create_internal()
        self.sess.run(tf.global_variables_initializer())
        self.train_scores = []
        np.set_printoptions(precision=6)

    def create_internal(self):
        td_error = self.better_value - self.critic_value
        self.value_loss = tf.reduce_mean(td_error ** 2)
        self.value_train_op = tf.train.AdamOptimizer(self.l_rate_critic).minimize(self.value_loss,
                                                                     var_list=self.value_weights)
        self.train_actor_op = tf.train.AdamOptimizer(self.l_rate).minimize(-tf.reduce_mean(self.value_for_train),
                                                                           var_list=self.weights)

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

    def update_target_weights(self, alpha=None):
        if alpha is None:
            alpha = self.tau
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
        self.update_target_weights(alpha=1.0)
        index_replay = 0
        iteration = 0
        episode = 0
        xp_replay_state = np.zeros(shape=(self.xp_size, self.env.get_observation_space()))
        xp_replay_next_state = np.zeros(shape=(self.xp_size, self.env.get_observation_space()))
        xp_replay_reward = np.zeros(shape=(self.xp_size,))
        xp_replay_action = np.zeros(shape=(self.xp_size, len(self.n_actions)))
        xp_replay_terminal = np.zeros(shape=(self.xp_size,))

        start_time = time.time()
        self.last_state = self.env.reset()
        env = self.env
        while True:
            if iteration <= self.random_steps:
                actions = env.env.action_space.sample()
            else:
                actions = self.act(env.features)
                actions += np.random.normal(0, scale=self.action_noise, size=actions.shape)

            env.step(actions)
            xp_replay_state[index_replay] = self.last_state.reshape(-1)
            xp_replay_next_state[index_replay] = env.features.reshape(-1)
            xp_replay_reward[index_replay] = env.reward
            xp_replay_action[index_replay] = actions.reshape(-1)
            xp_replay_terminal[index_replay] = env.done
            index_replay = (index_replay + 1) % self.xp_size
            if env.done or env.timestamp > self.timesteps_per_launch:
                episode += 1
                print("Episode #{}".format(episode), env.get_total_reward())
                self.train_scores.append(env.get_total_reward())
                env.reset()
            self.last_state = env.features
            if iteration % 1000 == 0:
                print("Iteration #{}".format(iteration))
                self.save(self.config[:-5])
            if iteration > self.random_steps:
                idxs = np.random.randint(np.min([xp_replay_state.shape[0], iteration]), size=self.batch_size)
                feed_dict = {
                    self.state_input: xp_replay_state[idxs],
                    self.next_state_input: xp_replay_next_state[idxs],
                    self.action_input: xp_replay_action[idxs],
                    self.reward_input: xp_replay_reward[idxs],
                    self.done_input: xp_replay_terminal[idxs]
                }

                self.sess.run([self.value_train_op], feed_dict)
                self.sess.run(self.train_actor_op, feed_dict)
                self.update_target_weights()
                weights = self.get_weights()
                for i, weight in enumerate(weights):
                    self.variables_server.set("weight_" + str(i), hlp.dump_object(weight))
                if iteration % self.test_every == 0:
                    print("Time to test!")
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
                    start_time = time.time()

            iteration += 1
