from typing import Tuple, Sequence

import numpy as np
from torch.utils.data import Dataset


class LSTM_Dataset(Dataset):
    def __init__(self, episodes: Sequence[Tuple[np.ndarray]], window_size, observations: Sequence[np.ndarray]=None):
        self.in_state = np.array([ep[0][:-1] for ep in episodes])  # n_episodes x episode_horizon x state_size
        self.actions = np.array([ep[1] for ep in episodes])  # n_episodes x episode_horizon x action_size
        if observations is None:
            observations = np.array([ep[0][1:] for ep in episodes])
        self.out_state = observations.copy()  # n_episodes x episode_horizon x state_size
        if not self.in_state.shape[:2] == self.actions.shape[:2] == self.out_state.shape[:2]:
            raise ValueError("Misaligned shapes")
        self.episode_horizon = self.in_state.shape[1]
        self.window_size = window_size
        windows_per_episode = self.episode_horizon - window_size + 1
        self.total_windows = self.episode_horizon * windows_per_episode

    def __len__(self):
        return self.total_windows

    def __getitem__(self, item):
        if item > self.total_windows:
            raise IndexError()
        windows_per_episode = self.episode_horizon - self.window_size + 1
        episode_ix = item // windows_per_episode
        offset = item % windows_per_episode
        in_states = self.in_state[episode_ix, offset:offset + self.window_size]  # window_size x state_size
        actions = self.actions[episode_ix, offset:offset + self.window_size]  # window_size x action_size
        out_state = self.out_state[episode_ix, offset + self.window_size - 1]  # state_size
        return in_states, actions, out_state
