from typing import Sequence, Tuple

import gym as gym
import numpy as np
import torch

from gpp.models.utilities import merge_episodes
from . import BaseModel
from .utilities import get_observations
from ..mdn import MDN, sample_gmm


class MDN_Model(BaseModel):
    def __init__(self, n_inputs, n_components, np_random=None):
        super().__init__(np_random=np_random)
        self.mdn = MDN(n_inputs, n_components)

    def train(self, episodes: Sequence[Tuple[np.ndarray]], epochs=None):
        if not epochs:
            raise ValueError("Missing required kwarg: epochs")
        state_size, action_size = episodes[0][0].shape[0], episodes[0][1].shape[0]
        self._check_input_sizes(action_size, state_size)
        x_array, y_array = merge_episodes(episodes)
        optimizer = torch.optim.Adam(self.mdn.parameters())
        x_variable = torch.from_numpy(x_array.astype(np.float32))
        y_variable = torch.from_numpy(y_array.astype(np.float32), requires_grad=False)
        # for

    def forward_sim(self, action_sequences: np.ndarray, env: gym.Env):
        curr_state = get_observations(env)
        n_sequences, T, action_size = action_sequences.shape
        state_size, = curr_state.shape
        self._check_input_sizes(state_size, action_size)
        outputs = np.zeros((n_sequences, T, state_size))
        action_sequences = torch.Tensor(action_sequences, requires_grad=False)
        curr_state = np.repeat([curr_state], n_sequences, axis=0)
        curr_state = torch.Tensor(curr_state, requires_grad=False)  # n_sequences x state_size
        for t in range(T):
            input_ = torch.cat([curr_state, action_sequences[:, t, :]], dim=1)
            pi, mu, sig2 = [x.data.numpy() for x in self.mdn.forward(input_)]
            curr_state = sample_gmm(pi, mu, sig2)
            outputs[:, t, :] = curr_state
        return outputs

    def _check_input_sizes(self, action_size, state_size):
        if action_size + state_size != self.mdn.n_inputs:
            raise ValueError(f"Actions and state have dimension {action_size} and {state_size} respectively"
                             f"but MDN was initialized to {self.mdn.n_inputs} inputs")
