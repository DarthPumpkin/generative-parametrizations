import gym
import numpy as np


class BaseModel:

    def __init__(self, np_random=None):
        if np_random is None:
            np_random = np.random.RandomState()
        self.np_random = np_random

    def forward_sim(self, action_sequences: np.ndarray, env: gym.Env):
        """
        Performs a forward pass on the given action sequences and returns the resulting states after each action
        :param action_sequences: the sequences of actions as a ndarray (n_sequences x horizon x action_space.size)
        :param env: the Gym environment used
        :return: the resulting states for each sequence of actions as a ndarray (action_sequences.shape)
        """
        raise NotImplementedError

    def train(self, *args, **kwargs):
        """Trains the model on a given dataset"""
        raise NotImplementedError
