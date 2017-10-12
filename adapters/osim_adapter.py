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
        self.features = np.array((self.env.reset(difficulty=difficulty))).reshape((1, -1))
        self.last_obs = np.zeros(shape=(1, 41))
        self.features = np.concatenate([self.features, self.last_obs], axis=1)
        self.done = False
        return self.features

    def get_action_space(self):
        space = [1] * 18
        return space

    def get_observation_space(self):
        return 41 * 2

    def step(self, actions):
        mean_possible = (np.array(self.env.action_space.low) + np.array(self.env.action_space.high))/2.
        actions = np.array(actions) + mean_possible
        actions *= (np.array(self.env.action_space.high) - np.array(self.env.action_space.low))
        actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
        obs, reward1, done, _ = self.env.step(actions)
        reward2 = 0
        if not done:
            obs, reward2, done, _ = self.env.step(actions)
        self.features = np.array(obs).reshape((1, -1))
        self.features = np.concatenate([self.features, self.features - self.last_obs], axis=1)
        self.last_obs = np.array(obs).reshape((1, -1))
        self.reward = reward1 + reward2
        self.total_reward += self.reward
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