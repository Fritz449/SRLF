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
        self.distributed = args['distributed']
        self.n_pre_tasks = args['n_pre_tasks']
        self.n_tests = args['n_tests']
        self.scale = args['scale']
        self.gamma = args['gamma']
        self.tau = args['tau']
        self.double = args['double']
        self.n_steps = args['n_steps']
        self.max_q_magnitude = args['max_magnitude']
        self.test_every = args['test_every']
        self.random_steps = args['random_steps']
        self.xp_size = args['xp_size']
        self.prioritized = args['prioritized']
        self.prior_alpha = args['prior_alpha']
        self.prior_beta = args['prior_beta']
        self.save_every = args.get('save_every', 500)
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
        self.action_input = tf.placeholder(tf.int32, shape=(None,))
        self.target_probs = tf.placeholder(tf.float32, shape=(self.batch_size, self.n_atoms))
        idx_batch = tf.reshape(tf.range(self.batch_size), [-1, 1])
        action_input = tf.reshape(self.action_input, [-1, 1])
        trainable_probs = tf.gather_nd(self.atom_probs, tf.concat([idx_batch, action_input], axis=1))
        cross_entropy = -self.target_probs*trainable_probs
        self.loss = tf.reduce_mean(cross_entropy, axis=1)
        self.importance_weights = tf.placeholder(tf.float32, shape=(None,))
        self.train_op = tf.train.AdamOptimizer(self.l_rate).minimize(tf.reduce_mean(self.importance_weights * self.loss), var_list=self.weights)


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

    def update_target_weights(self, alpha=None):
        if alpha is None:
            alpha = self.tau

        weights = self.get_weights()
        new_weights = self.get_target_weights()

        for i in range(len(weights)):
            new_weights[i] = new_weights[i] * (1 - alpha) + alpha * weights[i]

        self.set_target_weights(new_weights)

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
        index_replay = 0
        iteration = 0
        episode = 0
        idxs_range = np.arange(self.xp_size)
        xp_replay_state = np.zeros(shape=(self.xp_size, self.env.get_observation_space()))
        xp_replay_next_state = np.zeros(shape=(self.xp_size, self.env.get_observation_space()))
        xp_replay_reward = np.zeros(shape=(self.xp_size,))
        xp_replay_action = np.zeros(shape=(self.xp_size,))
        xp_replay_terminal = np.zeros(shape=(self.xp_size,))
        if self.prioritized:
            xp_replay_priority = np.zeros(shape=(self.xp_size,))
            self.max_prior = 1
        start_time = time.time()
        self.last_state = self.env.reset()
        discounts = self.gamma ** np.arange(self.n_steps)
        self.last_rewards = np.zeros(shape=(self.n_steps,))
        self.last_states = np.zeros(shape=(self.n_steps, self.n_features))
        self.last_actions = np.zeros(shape=(self.n_steps, ))
        buffer_index = 0
        env = self.env
        while True:
            if iteration <= self.random_steps:
                actions = env.env.action_space.sample()
            else:
                actions = self.act(env.features, exploration=True)
            self.last_states[buffer_index] = env.features.reshape(-1)
            self.last_actions[buffer_index] = actions
            env.step([actions])
            self.last_rewards[buffer_index] = env.reward
            buffer_index = (buffer_index + 1) % self.n_steps

            if env.timestamp >= self.n_steps:
                xp_replay_state[index_replay] = np.copy(self.last_states[buffer_index])
                xp_replay_next_state[index_replay] = env.features.reshape(-1)
                discounted_return = np.sum(discounts*self.last_rewards[np.roll(np.arange(self.n_steps), -(buffer_index))])
                xp_replay_reward[index_replay] = discounted_return
                xp_replay_action[index_replay] = self.last_actions[buffer_index]
                xp_replay_terminal[index_replay] = env.done
                if self.prioritized:
                    xp_replay_priority[index_replay] = self.max_prior
                index_replay = (index_replay + 1) % self.xp_size

            if env.done or env.timestamp > self.timesteps_per_launch:
                episode += 1
                print("Episode #{}".format(episode), env.get_total_reward())
                self.train_scores.append(env.get_total_reward())
                for i in range(1, self.n_steps):
                    buffer_index = (buffer_index + 1) % self.n_steps

                    xp_replay_state[index_replay] = np.copy(self.last_states[buffer_index])
                    xp_replay_next_state[index_replay] = env.features.reshape(-1)
                    discounted_return = np.sum(
                        discounts[:self.n_steps-i] * self.last_rewards[np.roll(np.arange(self.n_steps), -(buffer_index))[:self.n_steps-i]])
                    xp_replay_reward[index_replay] = discounted_return
                    xp_replay_action[index_replay] = self.last_actions[buffer_index]
                    xp_replay_terminal[index_replay] = env.done
                    index_replay = (index_replay + 1) % self.xp_size
                env.reset()
                self.last_rewards = np.zeros(shape=(self.n_steps,))
                self.last_states = np.zeros(shape=(self.n_steps, self.n_features))
                self.last_actions = np.zeros(shape=(self.n_steps,))
                buffer_index = 0

            self.last_state = env.features
            if iteration % 1000 == 0:
                print("Iteration #{}".format(iteration))
                self.save(self.config[:-5])

            if iteration > self.random_steps:
                if self.prioritized:
                    max_id = np.min([xp_replay_state.shape[0], iteration])
                    probs = xp_replay_priority[:max_id]/np.sum(xp_replay_priority[:max_id])
                    idxs = np.random.choice(idxs_range[:max_id], size=self.batch_size, p=probs)
                    importance_weights = (1/(max_id*probs[idxs]))**self.prior_beta
                else:
                    idxs = np.random.randint(np.min([xp_replay_state.shape[0], iteration]), size=self.batch_size)
                    importance_weights = np.ones(shape=(self.batch_size,))

                state_batch = xp_replay_state[idxs]
                next_state_batch = xp_replay_next_state[idxs]
                action_batch = xp_replay_action[idxs]
                reward_batch = xp_replay_reward[idxs]
                done_batch = xp_replay_terminal[idxs]

                feed_dict = {
                    self.state_input: state_batch,
                    self.next_state_input: next_state_batch,
                    self.action_input: action_batch,
                    self.importance_weights: importance_weights
                }
                target_atom_probs = self.sess.run(self.target_atom_probs, feed_dict)
                target_atom_probs = np.exp(target_atom_probs)

                if not self.double:
                    target_q_values = target_atom_probs * np.tile(np.arange(self.n_atoms).reshape((1, 1, self.n_atoms)), [self.batch_size, 1, 1])
                    target_q_values = np.sum(target_q_values, axis=2)
                    target_greedy_actions = np.argmax(target_q_values, axis=1).astype(np.int32).reshape((-1, 1))
                    target_probs = target_atom_probs[np.arange(self.batch_size).reshape((-1, 1)), target_greedy_actions]
                else:
                    feed_dict[self.state_input] = next_state_batch
                    atom_probs = self.sess.run(self.atom_probs, feed_dict)
                    atom_probs = np.exp(atom_probs)
                    q_values = atom_probs * np.tile(np.arange(self.n_atoms).reshape((1, 1, self.n_atoms)),
                                                                  [self.batch_size, 1, 1])
                    q_values = np.sum(q_values, axis=2)
                    greedy_actions = np.argmax(q_values, axis=1).astype(np.int32).reshape((-1, 1))
                    target_probs = target_atom_probs[np.arange(self.batch_size).reshape((-1, 1)), greedy_actions]
                    feed_dict[self.state_input] = state_batch

                atom_values = np.arange(self.n_atoms, dtype=np.float32).reshape((-1, self.n_atoms))
                atom_values = 2 * self.max_q_magnitude * (
                np.tile(atom_values, [self.batch_size, 1]) / (self.n_atoms - 1) - 0.5)
                atom_new_values = np.clip((self.gamma**self.n_steps) * atom_values * (1-done_batch).reshape(-1, 1) + reward_batch.reshape((-1, 1)),
                                                   - self.max_q_magnitude, self.max_q_magnitude)
                new_positions = ((atom_new_values / (2 * self.max_q_magnitude) + 0.5) * (self.n_atoms - 1)).reshape((-1))
                lower = np.floor(new_positions).astype(np.int32).reshape(-1)
                upper = np.floor(new_positions).astype(np.int32).reshape(-1) + 1

                final_target_probs = np.zeros(shape=(self.batch_size, self.n_atoms+1, self.n_atoms))
                final_target_probs[np.sort(np.tile(np.arange(self.batch_size), [self.n_atoms])), lower, np.tile(np.arange(self.n_atoms), [self.batch_size])] += (upper-new_positions) * target_probs.reshape((-1))
                final_target_probs[np.sort(np.tile(np.arange(self.batch_size), [self.n_atoms])), upper, np.tile(np.arange(self.n_atoms), [self.batch_size])] += (new_positions-lower) * target_probs.reshape((-1))

                final_target_probs = np.sum(final_target_probs, axis=2)[:, :-1]
                feed_dict[self.target_probs] = final_target_probs
                KLs = self.sess.run([self.loss, self.train_op], feed_dict)[0]
                if self.prioritized:
                    xp_replay_priority[idxs] = KLs ** self.prior_alpha
                self.update_target_weights()

                if iteration % self.test_every == 0:
                    weights = self.get_weights()
                    for i, weight in enumerate(weights):
                        self.variables_server.set("weight_" + str(i), hlp.dump_object(weight))
                    print("Time to test!")
                    if self.distributed:
                        weights = self.get_weights()
                        for i, weight in enumerate(weights):
                            self.variables_server.set("weight_" + str(i), hlp.dump_object(weight))
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
Time for iteration:        {tt}
Mean of features:          {means}
Std of features:           {stds}
-------------------------------------------------------------
                                    """.format(
                        means=means,
                        stds=stds,
                        test_scores=np.mean(total_rewards),
                        test_eplengths=np.mean(eplens),
                        max_test=np.max(total_rewards),
                        tt=time.time() - start_time
                    ))
                    start_time = time.time()
                    self.test_scores.append(np.mean(total_rewards))

            iteration += 1
            self.timestep += 1
