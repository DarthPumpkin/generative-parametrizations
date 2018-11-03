from tqdm import tqdm

import gym
import numpy as np
from gym.envs.classic_control import GaussianPendulumEnv

from gpp.models.utilities import get_observations


def test(env: gym.Env, controller_fun, test_runs: int, mass_mean, mass_stdev, episode_length=200, embed_knowledge=False,
         perfect_knowledge=False, seed=42):

    raw_env = env.unwrapped
    assert isinstance(raw_env, GaussianPendulumEnv)

    raw_env.configure(
        seed=seed,
        mass_mean=mass_mean,
        mass_stdev=mass_stdev,
        embed_knowledge=embed_knowledge,
        perfect_knowledge=perfect_knowledge,
        gym_env=env
    )

    # make test set (generate initial states)
    np.random.seed(seed)
    initial_states = np.random.uniform([0, -raw_env.max_speed], [2 * np.pi, raw_env.max_speed], size=(test_runs, 2))

    # run controller_fun on environment
    rewards = np.zeros(test_runs)
    masses = np.zeros(test_runs)
    for i in tqdm(range(test_runs)):
        env.reset()  # choose new mass and do other good things
        raw_env.state = initial_states[i]
        masses[i] = raw_env.physical_props[1]
        observations = get_observations(raw_env)
        for t in range(episode_length):
            action = controller_fun(observations)
            observations, reward, _, _ = env.step(action)
            rewards[i] += reward

    return masses, rewards
