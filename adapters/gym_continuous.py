import logging

import gym
import gym.spaces
import numpy as np

gym.undo_logger_setup()
gym_logger = logging.getLogger('gym')
gym_logger.setLevel(logging.ERROR)


class GymAdapterContinuous:
    def __init__(self, name):
        self.env = gym.make(name)
        self.reset()

    def step(self, actions):
        mean_possible = (np.array(self.env.action_space.low) + np.array(self.env.action_space.high))/2.
        actions = np.array(actions) - mean_possible
        actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
        obs, reward, done, _ = self.env.step(actions)
        obs = obs.reshape(-1)
        if type(obs[0]) == type(obs):
            obs[0] = obs[0][0]
        self.obs = obs.astype(np.float64)

        self.features = self.obs.reshape((1, -1))
        self.reward = reward/10
        self.total_reward += reward
        self.done = done
        self.timestamp += 1

    def reset(self):
        self.reward = 0
        self.total_reward = 0
        self.timestamp = 0.
        self.features = (self.env.reset()).reshape((1, -1)).astype(np.float64)
        obs = self.features.reshape(-1)
        if type(obs[0]) == type(obs):  # There is a strange bug here in some envs
            obs[0] = obs[0][0]
        self.obs = obs
        self.done = False
        return self.features

    def get_action_space(self):
        space = self.env.action_space
        return np.array(space.high) - np.array(space.low) + 1

    def get_observation_space(self):
        return self.env.observation_space.shape[0]

    def get_total_reward(self):
        return self.total_reward


class GymFactoryContinuous:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        return GymAdapterContinuous(self.name)

# Usage

# fact = GymFactoryContinuous('LunarLanderContinuous-v2')
# env = fact()
# while not env.done:
#     env.step(np.zeros(shape=env.get_action_space().shape[0]))
