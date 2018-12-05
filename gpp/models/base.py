from pathlib import Path
from typing import Sequence, Tuple

import gym
import numpy as np


class BaseModel:

    def __init__(self, np_random=None):
        if np_random is None:
            np_random = np.random.RandomState()
        self.np_random = np_random

    def forward_sim(self, action_sequences: np.ndarray, initial_state: np.ndarray, *args, **kwargs):
        """
        Performs a forward pass on the given action sequences and returns the resulting states after each action
        :param action_sequences: the sequences of actions as a ndarray (n_sequences x horizon x action_space.size)
        :param initial_state: the initial state (observations) of the Gym environment used
        :return: the resulting states for each sequence of actions as a ndarray (action_sequences.shape)
        """
        raise NotImplementedError

    def train(self, episodes: Sequence[Tuple[np.ndarray]]):
        """Trains the model on a given sequence of episodes
        :param episodes: sequence of episodes. Every episode is a tuple (state_sequence, action_sequence) of sizes (n
        + 1 x state_size) and (n x action_size), respectively"""
        raise NotImplementedError

    @staticmethod
    def load(file_path: Path):
        """TODO: Add documentation"""
        raise NotImplementedError

    def save(self, file_path: Path):
        """TODO: Add documentation"""
        raise NotImplementedError
