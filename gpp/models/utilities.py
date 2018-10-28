import gym
import gym.spaces
from collections import OrderedDict


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
