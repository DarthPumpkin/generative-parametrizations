from collections import OrderedDict

import gym
import gym.spaces
import numpy as np


def get_observation_space(env: gym.Env):
    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Box):
        return obs_space
    elif isinstance(obs_space, gym.spaces.Dict):
        subspaces = obs_space.spaces
        assert isinstance(subspaces, OrderedDict)
        box = subspaces['observation']
        assert isinstance(box, gym.spaces.Box)
        return box
    else:
        raise ValueError(f'Unknown observation space type ({obs_space})!')


def get_observations(env: gym.Env) -> np.ndarray:
    raise NotImplementedError


def merge_episodes(episodes):
    """Turns sequence of episodes into a single array for inputs and outputs, respectively. Episodes has same format
    as BaseModel.train"""
    state_size, action_size = episodes[0][0].shape[0], episodes[0][1].shape[0]
    cum_timesteps = np.cumsum([0] + [actions.shape[0] for (_, actions) in episodes])
    total_timesteps = cum_timesteps[-1]
    x_array = np.zeros((total_timesteps, state_size + action_size))
    y_array = np.zeros((total_timesteps, state_size))
    for i in range(len(episodes)):
        states, actions = episodes[i]
        sub_x = x_array[cum_timesteps[i]:cum_timesteps[i + 1]]  # this is a view, not a copy
        sub_y = y_array[cum_timesteps[i]:cum_timesteps[i + 1]]
        sub_x[:, :state_size] = states[:-1]
        sub_x[:, state_size:] = actions
        sub_y[:, :] = states[1:]
    return x_array, y_array
