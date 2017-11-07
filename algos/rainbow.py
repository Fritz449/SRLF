import tensorflow as tf
import numpy as np
import os
import sys
import random
import subprocess
from redis import Redis
import time
sys.path.append(os.path.realpath(".."))
from helpers.layers import denselayer
from models.rainbow_network import RainbowNetwork
import helpers.utils as hlp

class RainbowTrainer(RainbowNetwork):
    def __init__(self, sess, args):
        RainbowNetwork.__init__(self, sess, args)
        self.sess = sess
        self.config = args['config']
        self.env = args['environment']
        self.l_rate = args['learning_rate']
        self.timesteps_per_launch = args['max_pathlength']
        self.n_workers = args['n_workers']
        self.n_pre_tasks = args['n_pre_tasks']
        self.n_tests = args['n_tests']
        self.scale = args['scale']
        self.gamma = args['gamma']
        self.tau = args['tau']
        self.max_q_magnitude = args['max_magnitude']
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
        self.action_input = tf.placeholder(tf.int32, shape=(None,))
        self.target_probs = tf.placeholder(tf.float32, shape=(self.batch_size, self.n_atoms))
        idx_batch = tf.reshape(tf.range(self.batch_size), [-1, 1])
        action_input = tf.reshape(self.action_input, [-1, 1])
        trainable_probs = tf.gather_nd(self.atom_probs, tf.concat([idx_batch, action_input], axis=1))
        cross_entropy = -self.target_probs*trainable_probs
        self.loss = tf.reduce_mean(cross_entropy,  axis=1)
        self.train_op = tf.train.AdamOptimizer(self.l_rate).minimize(self.loss, var_list=self.weights)


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

    def update_target_weights(self, alpha=None):
        if alpha is None:
            alpha = self.tau

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
        xp_replay_action = np.zeros(shape=(self.xp_size,))
        xp_replay_terminal = np.zeros(shape=(self.xp_size,))

        start_time = time.time()
        self.last_state = self.env.reset()
        env = self.env
        while True:
            if iteration <= self.random_steps:
                actions = env.env.action_space.sample()
            else:
                actions = self.act(env.features, exploration=True)

            env.step([actions])
            xp_replay_state[index_replay] = self.last_state.reshape(-1)
            xp_replay_next_state[index_replay] = env.features.reshape(-1)
            xp_replay_reward[index_replay] = env.reward
            xp_replay_action[index_replay] = actions
            xp_replay_terminal[index_replay] = env.done
            index_replay = (index_replay + 1) % self.xp_size
            if env.done or env.timestamp > self.timesteps_per_launch:
                episode += 1
                print("Episode #{}".format(episode), env.get_total_reward())
                self.train_scores.append(env.get_total_reward())
                np.save('train_scores', self.train_scores)
                env.reset()
            self.last_state = env.features
            if iteration % 1000 == 0:
                print("Iteration #{}".format(iteration))
                self.save(self.config[:-5])

            if iteration > self.random_steps:
                start_time = time.time()
                idxs = np.random.randint(np.min([xp_replay_state.shape[0], iteration]), size=self.batch_size)

                state_batch = xp_replay_state[idxs]
                next_state_batch = xp_replay_next_state[idxs]
                action_batch = xp_replay_action[idxs]
                reward_batch = xp_replay_reward[idxs]
                done_batch = xp_replay_terminal[idxs]
                feed_dict = {
                    self.state_input: state_batch,
                    self.next_state_input: next_state_batch,
                    self.action_input: action_batch,
                }
                target_atom_probs = self.sess.run(self.target_atom_probs, feed_dict)
                target_atom_probs = np.exp(target_atom_probs)
                target_q_values = target_atom_probs * np.tile(np.arange(self.n_atoms).reshape((1, 1, self.n_atoms)), [self.batch_size, 1, 1])
                target_q_values = np.sum(target_q_values, axis=2)
                target_greedy_actions = np.argmax(target_q_values, axis=1).astype(np.int32).reshape((-1, 1))
                target_probs = target_atom_probs[np.arange(self.batch_size).reshape((-1, 1)), target_greedy_actions]
                atom_values = np.arange(self.n_atoms, dtype=np.float32).reshape((-1, self.n_atoms))
                atom_values = 2 * self.max_q_magnitude * (
                np.tile(atom_values, [self.batch_size, 1]) / (self.n_atoms - 1) - 0.5)
                atom_new_values = np.clip(self.gamma * atom_values * (1-done_batch).reshape(-1, 1) + reward_batch.reshape((-1, 1)),
                                                   - self.max_q_magnitude, self.max_q_magnitude)
                new_positions = ((atom_new_values / (2 * self.max_q_magnitude) + 0.5) * (self.n_atoms - 1)).reshape((-1))
                lower = np.floor(new_positions).astype(np.int32).reshape(-1)
                upper = np.floor(new_positions).astype(np.int32).reshape(-1) + 1

                final_target_probs = np.zeros(shape=(self.batch_size, self.n_atoms+1, self.n_atoms))
                final_target_probs[np.sort(np.tile(np.arange(self.batch_size), [self.n_atoms])), lower, np.tile(np.arange(self.n_atoms), [self.batch_size])] += (upper-new_positions) * target_probs.reshape((-1))
                final_target_probs[np.sort(np.tile(np.arange(self.batch_size), [self.n_atoms])), upper, np.tile(np.arange(self.n_atoms), [self.batch_size])] += (new_positions-lower) * target_probs.reshape((-1))

                final_target_probs = np.sum(final_target_probs, axis=2)[:, :-1]
                feed_dict[self.target_probs] = final_target_probs
                self.sess.run(self.train_op, feed_dict)
                self.update_target_weights()

                if iteration % self.test_every == 0:
                    weights = self.get_weights()
                    for i, weight in enumerate(weights):
                        self.variables_server.set("weight_" + str(i), hlp.dump_object(weight))
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
            self.timestep += 1
