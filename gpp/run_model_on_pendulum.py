import gym
import numpy as np
from gym.envs.classic_control import GaussianPendulumEnv

from gpp.models.utilities import get_observations


def test(controller_fun, test_runs: int, mass_mean, mass_stdev, episode_length=200, embed_knowledge=False,
         perfect_knowledge=False):
    # initialize environment
    tm_env = gym.make('GaussianPendulum-v0')
    raw_env = tm_env.unwrapped
    assert isinstance(raw_env, GaussianPendulumEnv)

    raw_env.configure(
        seed=42,
        mass_mean=mass_mean,
        mass_stdev=mass_stdev,
        embed_knowledge=embed_knowledge,
        perfect_knowledge=perfect_knowledge,
        gym_env=tm_env
    )

    # make test set (generate initial states)
    np.random.seed(42)
    initial_states = np.random.uniform([0, -raw_env.max_speed], [2 * np.pi, raw_env.max_speed], size=(test_runs, 2))

    # run controller_fun on environment
    rewards = np.zeros(test_runs)
    masses = np.zeros(test_runs)
    for i in range(test_runs):
        raw_env.reset()  # choose new mass and do other good things
        raw_env.state = initial_states[i]
        masses[i] = raw_env.physical_props[1]
        observerations = get_observations(raw_env)
        for t in range(episode_length):
            action = controller_fun(observerations)
            observerations, reward, _, _ = tm_env.step(action)
            rewards[i] += reward

    return masses, rewards
