import gym
import numpy as np
from gym.envs.classic_control import PendulumEnv

import _gpp
from gpp.reward_functions import RewardFunction


def test_pendulum_env():

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
        assert np.allclose(reward_a, reward_b)

    rewards_b = reward_fn(all_obs, actions=all_actions)
    assert np.allclose(rewards_a, rewards_b)

    print('==> test_pendulum_env PASSED')


if __name__ == '__main__':
    test_pendulum_env()
    print('==> all tests PASSED')
