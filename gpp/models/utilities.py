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
    if hasattr(env, 'unwrapped'):
        env = env.unwrapped
    if hasattr(env, 'get_obs'):
        res = env.get_obs()
    elif hasattr(env, '_get_obs'):
        res = env._get_obs()
    else:
        raise ValueError(f'Can\'t get observations from environment {type(env)}!')
    if isinstance(res, dict):
        return res['observation']
    else:
        return res


def merge_episodes(episodes, targets=None):
    """Turns sequence of episodes into a single array for inputs and outputs, respectively. Episodes has same format
    as BaseModel.train"""
    in_state_size, action_size = episodes[0][0].shape[1], episodes[0][1].shape[1]
    cum_timesteps = np.cumsum([0] + [actions.shape[0] for (_, actions) in episodes])
    total_timesteps = cum_timesteps[-1]
    x_array = np.zeros((total_timesteps, in_state_size + action_size))

    if targets is None:
        out_state_size = in_state_size
    else:
        out_state_size = targets.shape[2]

    y_array = np.zeros((total_timesteps, out_state_size))

    for i in range(len(episodes)):
        states, actions = episodes[i]
        sub_x = x_array[cum_timesteps[i]:cum_timesteps[i + 1]]  # this is a view, not a copy
        sub_y = y_array[cum_timesteps[i]:cum_timesteps[i + 1]]
        sub_x[:, :in_state_size] = states[:-1]
        sub_x[:, in_state_size:] = actions

        if targets is None:
            sub_y[:, :] = states[1:]
        else:
            sub_y[:, :] = targets[i].copy()

    return x_array, y_array
