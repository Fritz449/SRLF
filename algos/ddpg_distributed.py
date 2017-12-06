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
        self.step_delay = args['step_delay']
        self.xp_size = args['xp_size']
        self.save_every = args.get('save_every', 1)
        self.clip_error = args.get('clip_error', 10.)
        self.batch_size = args['batch_size']
        self.sums = self.sumsqrs = self.sumtime = 0
        self.timestep = 0
        self.create_internal()
        self.sess.run(tf.global_variables_initializer())

        self.train_scores = []
        self.test_scores = []
        np.set_printoptions(precision=6)

        # Worker parameters:
        self.id_worker = args['id_worker']
        self.test_mode = args['test_mode']

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

        for i, w in enumerate(tf.global_variables()):
            np.save(directory + 'weight_{}'.format(i), self.sess.run(w))

        if self.scale:
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
            weights = [np.zeros(shape=w.get_shape()) for w in self.weights]
            for i, w in enumerate(tf.global_variables()):
                weights[i] = np.load(directory + 'weight_{}.npy'.format(i))
            self.set_weights(weights)

            if self.scale:
                self.sums = np.load(directory + 'sums.npy')
                self.sumsqrs = np.load(directory + 'sumsquares.npy')
                self.sumtime = np.load(directory + 'sumtime.npy')

            self.timestep = np.load(directory + 'timestep.npy')[0]
            self.train_scores = np.load(directory + 'train_scores.npy').tolist()
            self.test_scores = np.load(directory + 'test_scores.npy').tolist()

            print("Agent successfully loaded from folder {}".format(directory))
        except:
            print("Something is wrong, loading failed")

    def load_weights_from_redis(self):
        weights = [hlp.load_object(self.variables_server.get("weight_{}".format(i))) for i in range(len(self.weights))]
        self.set_weights(weights)

    def work(self):
        self.variables_server = Redis(port=12000)
        if self.scale != 'off':
            try:
                means = hlp.load_object(self.variables_server.get("means"))
                stds = hlp.load_object(self.variables_server.get("stds"))
                self.sess.run(self.norm_set_op, feed_dict=dict(zip(self.norm_phs, [means, stds])))
            except:
                pass
        try:
            weights = [hlp.load_object(self.variables_server.get("weight_{}".format(i))) for i in
                       range(len(self.weights))]
            self.set_weights(weights)
        except:
            pass
        env = self.env
        local_iteration = 0

        while True:
            st = time.time()
            self.last_state = env.reset()
            self.load_weights_from_redis()
            while not env.done and env.timestamp < self.timesteps_per_launch:
                if local_iteration * self.n_workers <= self.random_steps:
                    actions = env.env.action_space.sample()
                else:
                    actions = self.act(env.features)
                    actions += np.random.normal(0, scale=self.action_noise, size=actions.shape)
                env.step(actions)
                transition = hlp.dump_object([self.last_state, env.reward, actions, env.features, env.done])
                time.sleep(self.step_delay)
                self.variables_server.lpush('transitions', transition)
                self.last_state = env.features
                if time.time() - st > 3:
                    st = time.time()
                    self.load_weights_from_redis()
                local_iteration += 1
            print("Episode reward: {}".format(env.get_total_reward()), "Length: {}".format(env.timestamp))
            if self.variables_server.llen('transitions') > self.xp_size:
                self.variables_server.ltrim('transitions', 1, self.xp_size)

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
        if self.test_mode:
            n_tasks = self.n_tests
        else:
            n_tasks = self.n_pre_tasks
        i_task = 0
        paths = []
        while i_task < n_tasks:
            path = {}
            rewards = []
            sums = np.zeros((1, env.get_observation_space()))
            sumsqrs = np.zeros(sums.shape)

            env.reset()
            while not env.done and env.timestamp < self.timesteps_per_launch:
                sums += env.features
                sumsqrs += np.square(env.features)
                if not self.test_mode:
                    actions = self.act(env.features, exploration=True)
                else:
                    actions = self.act(env.features, exploration=False)
                env.step(actions)
                rewards.append(env.reward)

            path["rewards"] = rewards
            path["sumobs"] = sums
            path["sumsqrobs"] = sumsqrs
            path["total"] = env.get_total_reward()
            paths.append(path)
            i_task += 1

        variables_server.set("paths_{}".format(self.id_worker), hlp.dump_object(paths))

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
        means = "-"
        stds = "-"
        if self.scale != 'off':
            if self.timestep == 0:
                print("Time to measure features!")
                worker_args = \
                    {
                        'config': self.config,
                        'test_mode': False,
                    }
                hlp.launch_workers(worker_args, self.n_workers)
                paths = []
                for i in range(self.n_workers):
                    paths += hlp.load_object(self.variables_server.get("paths_{}".format(i)))

                for path in paths:
                    self.sums += path["sumobs"]
                    self.sumsqrs += path["sumsqrobs"]
                    self.sumtime += len(path["rewards"])

            stds = np.sqrt((self.sumsqrs - np.square(self.sums) / self.sumtime) / (self.sumtime - 1))
            means = self.sums / self.sumtime
            print("Init means: {}".format(means))
            print("Init stds: {}".format(stds))
            self.variables_server.set("means", hlp.dump_object(means))
            self.variables_server.set("stds", hlp.dump_object(stds))
            self.sess.run(self.norm_set_op, feed_dict=dict(zip(self.norm_phs, [means, stds])))
        print("Let's go!")
        self.update_target_weights(alpha=1.0)

        weights = self.get_weights()
        for i, weight in enumerate(weights):
            self.variables_server.set("weight_" + str(i), hlp.dump_object(weight))
        worker_args = \
            {
                'config': self.config,
                'test_mode': False,
            }
        hlp.launch_workers(worker_args, self.n_workers, command='work', wait=False)

        self.variables_server.ltrim('transitions', 0, 0)

        time.sleep(5)
        iteration = 0

        start_time = time.time()
        max_idx = self.variables_server.llen('transitions')
        while True:
            obs_batch = []
            next_obs_batch = []
            done_batch = []
            reward_batch = []
            actions_batch = []
            if iteration % 500 == 0 and max_idx < self.xp_size:
                max_idx = self.variables_server.llen('transitions')
            idxs = np.random.randint(np.min([self.xp_size, max_idx]), size=self.batch_size)
            transitions = []
            for i in range(self.batch_size):
                transitions.append(hlp.load_object(self.variables_server.lindex('transitions', idxs[i])))
            for transition in transitions:
                obs_batch.append(transition[0])
                reward_batch.append(transition[1])
                actions_batch.append(transition[2])
                next_obs_batch.append(transition[3])
                done_batch.append(transition[4])
            feed_dict = {
                self.state_input: np.concatenate(obs_batch, axis=0),
                self.next_state_input: np.concatenate(next_obs_batch, axis=0),
                self.action_input: np.array(actions_batch),
                self.reward_input: np.array(reward_batch),
                self.done_input: np.array(done_batch)
            }

            self.sess.run(self.value_train_op, feed_dict)
            self.sess.run(self.train_actor_op, feed_dict)
            self.update_target_weights()
            weights = self.get_weights()
            for i, weight in enumerate(weights):
                self.variables_server.set("weight_" + str(i), hlp.dump_object(weight))
            if iteration % 1000 == 0:
                print("Iteration #{}".format(iteration))
                self.save(self.config[:-5])
            if iteration % self.test_every == 0:
                print("Time for testing!")
                worker_args = \
                    {
                        'config': self.config,
                        'test_mode': True,
                    }
                hlp.launch_workers(worker_args, self.n_workers)
                paths = []
                for i in range(self.n_workers):
                    paths += hlp.load_object(self.variables_server.get("paths_{}".format(i)))

                total_rewards = np.array([path["total"] for path in paths])
                eplens = np.array([len(path["rewards"]) for path in paths])

                if self.scale == 'full':
                    stds = np.sqrt((self.sumsqrs - np.square(self.sums) / self.sumtime) / (self.sumtime - 1))
                    means = self.sums / self.sumtime
                    self.variables_server.set("means", hlp.dump_object(means))
                    self.variables_server.set("stds", hlp.dump_object(stds))
                    self.sess.run(self.norm_set_op, feed_dict=dict(zip(self.norm_phs, [means, stds])))

                print("""
                -------------------------------------------------------------
                Mean test score:           {test_scores}
                Mean test episode length:  {test_eplengths}
                Max test score:            {max_test}
                Mean of features:          {means}
                Std of features:           {stds}
                Time for iteration:        {tt}
                -------------------------------------------------------------
                                """.format(
                    means=means,
                    stds=stds,
                    test_scores=np.mean(total_rewards),
                    test_eplengths=np.mean(eplens),
                    max_test=np.max(total_rewards),
                    tt=time.time() - start_time
                ))
                self.test_scores.append(np.mean(total_rewards))

            iteration += 1
            self.timestep += 1
