import pickle
from pathlib import Path

from tqdm import tqdm
import gym
from torch.utils.data import Dataset
from .models.utilities import get_observation_space, get_observations
import numpy as np


class EnvDataset(Dataset):

    def __init__(self, env: gym.Env):
        super(EnvDataset, self).__init__()
        self.data = None
        self.env = env
        self._raw_env = env.unwrapped

    def __getitem__(self, index):
        if self.data is None:
            raise ValueError('You must call generate() first!')
        return self.data[index]

    def __len__(self):
        return self.episodes

    @property
    def episodes(self):
        if self.data is None:
            return 0
        return len(self.data)

    @property
    def episode_length(self):
        if self.episodes == 0:
            return 0
        return self.data[0][1].shape[0]

    @property
    def state_shape(self):
        if self.episodes == 0 or self.episode_length == 0:
            return 0
        return self.data[0][0][0].shape

    def generate(self, episodes: int, episode_length: int, strategy=None, strategy_period=1):
        env, raw_env = self.env, self._raw_env
        observation_space = get_observation_space(env)

        def rand_strategy(*args):
            return raw_env.action_space.sample()

        strategy = strategy or rand_strategy
        data = []
        for e in tqdm(range(episodes)):
            env.reset()
            obs = get_observations(env)
            actions = np.zeros((episode_length,) + raw_env.action_space.shape)
            states = np.zeros((episode_length + 1,) + observation_space.shape)
            states[0] = obs
            for s in range(episode_length):
                if s % strategy_period == 0:
                    a = strategy(raw_env, obs)
                _, rewards, dones, info = env.step(a)
                obs = get_observations(env)
                states[s + 1] = obs
                actions[s] = a
            data.append((states, actions))
        self.data = data
        return data

    def save(self, file_path: Path):
        file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self.data, f)

    def load(self, file_path: Path):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
