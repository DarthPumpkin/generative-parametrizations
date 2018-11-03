from pathlib import Path
from typing import Sequence, Tuple

import gym as gym
import numpy as np
import torch
from torch import tensor

from gpp.models.utilities import merge_episodes
from . import BaseModel
from .utilities import get_observations
from ..mdn import MDN, sample_gmm, mdn_loss


class MDN_Model(BaseModel):

    def __init__(self, n_inputs, n_outputs, n_components, np_random=None, device=torch.device("cpu")):
        super().__init__(np_random=np_random)
        self.mdn = MDN(n_inputs, n_outputs, n_components)
        self.mdn = self.mdn.to(device)
        self.device = device

    def train(self, episodes: Sequence[Tuple[np.ndarray]], epochs: int=None, batch_size: int=None):

        if not epochs:
            raise ValueError("Missing required kwarg: epochs")

        if not batch_size:
            batch_size = int(1e20)

        state_size, action_size = episodes[0][0].shape[1], episodes[0][1].shape[1]
        self._check_input_sizes(action_size, state_size)
        x_array, y_array = merge_episodes(episodes)
        optimizer = torch.optim.Adam(self.mdn.parameters())

        n_batches = int(np.ceil(1.0 * x_array.shape[0] / batch_size))
        losses = np.zeros(epochs)

        for t in range(epochs):

            for batch in range(n_batches):

                idx_lb = batch_size * batch
                idx_up = min(batch_size * (batch + 1), x_array.shape[0])
                x_batch = x_array[idx_lb:idx_up]
                y_batch = y_array[idx_lb:idx_up]

                x_variable = torch.from_numpy(x_batch.astype(np.float32)).to(self.device)  # type: tensor.Tensor
                y_variable = torch.from_numpy(y_batch.astype(np.float32)).to(self.device)  # type: tensor.Tensor
                assert hasattr(y_variable, 'requires_grad')
                y_variable.requires_grad = False

                pi, mu, sig2 = self.mdn(x_variable)
                loss = mdn_loss(pi, mu, sig2, y_variable)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses[t] += 1.0 * loss.data.item() / x_batch.shape[0]

            print(t, losses[t])

        return losses

    def forward_sim(self, action_sequences: np.ndarray, initial_state: np.ndarray):
        curr_state = initial_state
        n_sequences, T, action_size = action_sequences.shape
        state_size, = curr_state.shape
        self._check_input_sizes(state_size, action_size)
        outputs = np.zeros((n_sequences, T, state_size))
        action_sequences = torch.Tensor(action_sequences).to(self.device)
        curr_state = np.repeat([curr_state], n_sequences, axis=0)
        curr_state = torch.Tensor(curr_state).to(self.device)  # n_sequences x state_size

        with torch.no_grad():
            for t in range(T):
                input_ = torch.cat([curr_state, action_sequences[:, t, :]], dim=1)
                pi, mu, sig2 = [x.cpu().data.numpy() for x in self.mdn.forward(input_)]
                curr_state = sample_gmm(pi, mu, sig2)
                outputs[:, t, :] = curr_state
                curr_state = torch.Tensor(curr_state).to(self.device)

        return outputs

    def save(self, file_path: Path):
        file_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.mdn.cpu().state_dict(), file_path)
        if self.device is not None and self.device.type != 'cpu':
            self.mdn.cuda()

    def load(self, file_path: Path):
        state_dict = torch.load(file_path)
        self.mdn.load_state_dict(state_dict)
        if self.device is not None:
            self.mdn = self.mdn.to(self.device)

    def _check_input_sizes(self, action_size, state_size):
        if action_size + state_size != self.mdn.n_inputs:
            raise ValueError(f"Actions and state have dimension {action_size} and {state_size} respectively "
                             f"but MDN was initialized to {self.mdn.n_inputs} inputs")
