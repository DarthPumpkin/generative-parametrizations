import pickle
from pathlib import Path

import gym
from torch.utils.data import Dataset
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
        return self.data[0][1].size

    def generate(self, episodes: int, episode_length: int):
        env, raw_env = self.env, self._raw_env
        data = []
        for e in range(episodes):
            obs = env.reset()
            actions = np.zeros((episode_length,) + raw_env.action_space.shape)
            states = np.zeros((episode_length + 1,) + raw_env.observation_space.shape)
            states[0] = obs
            for s in range(episode_length):
                rand_action = raw_env.action_space.sample()
                obs, rewards, dones, info = env.step(rand_action)
                states[s + 1] = obs
                actions[s] = rand_action
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
