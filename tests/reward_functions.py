import gym
import numpy as np
import unittest
from gym.envs.classic_control import PendulumEnv
from gym.envs.robotics import FetchEnv

import _gpp
from gpp.reward_functions import RewardFunction


class TestRewardFunction(unittest.TestCase):

    def test_pendulum_env(self):

        env = gym.make('Pendulum-v0')
        raw_env = env.unwrapped # type: PendulumEnv
        reward_fn = RewardFunction(env)

        obs = env.reset()
        n_trials = 10000

        rewards_a = np.zeros(n_trials)
        all_actions = np.zeros((n_trials,) + raw_env.action_space.shape)
        all_obs = np.zeros((n_trials,) + raw_env.observation_space.shape)

        for i in range(n_trials):
            action = raw_env.action_space.sample()
            all_obs[i] = obs.copy()
            all_actions[i] = action.copy()

            reward_b = reward_fn(obs, actions=action)
            obs, reward_a, _, _ = env.step(action)
            rewards_a[i] = reward_a
            self.assertTrue(np.allclose(reward_a, reward_b))

        rewards_b = reward_fn(all_obs, actions=all_actions)
        self.assertTrue(np.allclose(rewards_a, rewards_b))

    def test_fetch_env(self):

        cases = [
            'FetchReach-v1',
            'FetchSlideDense-v1',
            'FetchPickAndPlaceSphere-v1',
            'FetchPush-v1'
        ]

        for c in cases:

            env = gym.make(c)
            raw_env = env.unwrapped  # type: FetchEnv
            reward_fn = RewardFunction(env)

            env.reset()
            n_trials = 2000

            obs_space_shape = raw_env.observation_space.spaces['observation'].shape

            rewards_a = np.zeros(n_trials)
            all_actions = np.zeros((n_trials,) + raw_env.action_space.shape)
            all_obs = np.zeros((n_trials,) + obs_space_shape)
            goal = None

            for i in range(n_trials):
                action = raw_env.action_space.sample()
                obs, reward_a, _, _ = env.step(action)

                goal = obs['desired_goal']
                obs = obs['observation']
                all_obs[i] = obs.copy()
                all_actions[i] = action.copy()

                reward_b = reward_fn(obs, goal=goal)
                rewards_a[i] = reward_a
                self.assertTrue(np.allclose(reward_a, reward_b))

            rewards_b = reward_fn(all_obs, goal=goal)
            self.assertTrue(np.allclose(rewards_a, rewards_b))


if __name__ == '__main__':
    unittest.main(verbosity=2)
