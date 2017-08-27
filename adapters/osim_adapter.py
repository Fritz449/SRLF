from osim.env import RunEnv
import numpy as np

class OsimAdapter:
    def __init__(self):
        self.env = RunEnv(visualize = False)
        self.reset()

    def reset(self, difficulty=2):
        self.reward = 0
        self.total_reward = 0
        self.timestamp = 0.
        self.features = (self.env.reset(difficulty=difficulty)).reshape((1, -1))
        self.done = False
        return self.features

    def get_action_space(self):
        space = [1] * 18
        return space

    def get_observation_space(self):
        return 41

    def step(self, actions):
        mean_possible = (np.array(self.env.action_space.low) + np.array(self.env.action_space.high))/2.
        actions = np.array(actions) - mean_possible
        actions *= 2/(np.array(self.env.action_space.high) - np.array(self.env.action_space.low))
        actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)

        obs, reward, done, _ = self.env.step(actions)
        self.features = obs.reshape((1, -1))
        self.reward = reward
        self.total_reward += reward
        self.done = done
        self.timestamp += 1

    def get_total_reward(self):
        return self.total_reward


class OsimFactory:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return OsimAdapter()


# Usage
#
# fact = OsimFactory()
# env = fact()
# while not env.done:
#     env.step([1e-4]*18)