from tqdm import tqdm
import pandas as pd
import gym
import numpy as np
from gym.envs.classic_control import GaussianPendulumEnv

from gpp.models.utilities import get_observations


def generate_test_dataset(seed=42, episodes=300) -> pd.DataFrame:

    env = gym.make('GaussianPendulum-v0')
    raw_env = env.unwrapped
    assert isinstance(raw_env, GaussianPendulumEnv)

    raw_env.configure(
        seed=seed,
        mass_mean=(0.050, 1.300),
        mass_stdev=0.0,
        embed_knowledge=True,
        perfect_knowledge=False,
        gym_env=env
    )

    np_random = np.random.RandomState(seed=seed)
    initial_states = np_random.uniform([0, -raw_env.max_speed], [2 * np.pi, raw_env.max_speed], size=(episodes, 2))

    data = []
    for e in range(episodes):
        env.reset()
        data.append(dict(
            mass_distr_params=raw_env.mass_distr_params,
            sampled_mass=raw_env.physical_props[1],
            initial_state=initial_states[e]
        ))

    return pd.DataFrame(data)


def test(env: gym.Env, controller_fun, test_data: pd.DataFrame, episode_length=200, embed_knowledge=False,
         perfect_knowledge=False):

    raw_env = env.unwrapped
    assert isinstance(raw_env, GaussianPendulumEnv)

    initial_states = test_data.initial_state
    mass_distr_params = test_data.mass_distr_params
    sampled_masses = test_data.sampled_mass
    episodes = len(test_data)
    pd_index = list(test_data.index)

    # configure the environment
    raw_env.configure(
        mass_mean=np.mean(sampled_masses),
        mass_stdev=np.std(sampled_masses),
        embed_knowledge=embed_knowledge,
        perfect_knowledge=perfect_knowledge,
        gym_env=env
    )

    # Note that mass mean and stdev passed in configure() are not actually used by the environment itself,
    # as we will set our predefined values of masses. It's still useful to have some numbers in there
    # because they will be used to define the bounds of the observation space.

    # run controller_fun on environment
    rewards = np.zeros(episodes)
    for i in tqdm(range(episodes)):
        pdi = pd_index[i]

        # reset the environment to ensure that everything is clean
        env.reset()

        # impose initial state
        raw_env.state = initial_states[pdi]

        # impose sampled mass
        raw_env.sampled_mass = sampled_masses[pdi]

        # inform env about the original distribution used
        raw_env.mass_distr_params = mass_distr_params[pdi]

        observations = get_observations(raw_env)
        for t in range(episode_length):
            action = controller_fun(observations)
            observations, reward, _, _ = env.step(action)
            rewards[i] += reward

    return rewards
