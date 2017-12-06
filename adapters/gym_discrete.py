import logging

import gym
import gym.spaces
import numpy as np

gym.undo_logger_setup()
gym_logger = logging.getLogger('gym')
gym_logger.setLevel(logging.ERROR)


class GymAdapterDiscrete:
    def __init__(self, name):
        self.env = gym.make(name)
        self.reset()

    def step(self, actions):
        if type(actions) == list or type(actions) == np.ndarray:
            actions = actions[0]
        obs, reward, done, _ = self.env.step(actions)
        self.features = obs.reshape((1, -1))
        self.reward = np.clip(reward, -1, 1)
        self.total_reward += reward
        self.done = done
        self.timestamp += 1

    def reset(self):
        self.reward = 0
        self.total_reward = 0
        self.timestamp = 0.
        self.features = (self.env.reset()).reshape((1, -1))
        self.done = False
        return self.features

    def get_action_space(self):
        space = self.env.action_space
        return [space.n]

    def get_observation_space(self):
        return self.env.observation_space.shape[0]

    def get_total_reward(self):
        return self.total_reward


class GymFactoryDiscrete:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        return GymAdapterDiscrete(self.name)


# Usage
#
# fact = GymFactoryDiscrete('CartPole-v0')
# env = fact()
# while not env.done:
#     env.step(np.random.randint(env.get_action_space()))
