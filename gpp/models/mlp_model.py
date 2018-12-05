import pickle
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from torch import tensor, nn
from sklearn.preprocessing import StandardScaler

from gpp.models.utilities import merge_episodes
from . import BaseModel


class _MlpNetwork(nn.Module):

    def __init__(self, n_inputs: int, n_outputs: int, hidden_units: list):
        super(_MlpNetwork, self).__init__()

        sizes = [n_inputs] + hidden_units

        self.h = nn.Sequential(*[nn.Sequential(
            nn.Linear(sizes[i], sizes[i + 1]),
            nn.Tanh()
        ) for i in range(len(sizes) - 1)])

        self.h_to_out = nn.Linear(hidden_units[-1], n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.h_to_out(self.h(x))


class MlpModel(BaseModel):

    def __init__(self, n_inputs, n_outputs, hidden_units=None, np_random=None, device=torch.device("cpu")):
        super().__init__(np_random=np_random)

        hidden_units = list(hidden_units or [20])
        self.hidden_units = hidden_units
        self.n_inputs, self.n_outputs = n_inputs, n_outputs
        self.loss_function = nn.MSELoss()
        self.mlp = _MlpNetwork(n_inputs, n_outputs, hidden_units)
        self.device = device
        self._state_scaler = None
        self._action_scaler = None

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self.mlp = self.mlp.to(value)
        self._device = value

    @staticmethod
    def load(file_path: Path, device=torch.device("cpu")):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
            if not isinstance(model, MlpModel):
                raise ValueError(f'Cannot load object of type {type(model)}!')
            model.device = device
            return model

    def save(self, file_path: Path):
        file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def train(self, episodes: Sequence[Tuple[np.ndarray]], epochs: int=None, batch_size: int=None, epoch_callback=None,
              scale_data=False, shuffle_data=False):

        if not epochs:
            raise ValueError("Missing required kwarg: epochs")

        if not batch_size:
            batch_size = int(1e20)

        state_size, action_size = episodes[0][0].shape[1], episodes[0][1].shape[1]
        self._check_input_sizes(action_size, state_size)
        x_array, y_array = merge_episodes(episodes)
        optimizer = torch.optim.Adam(self.mlp.parameters())
        # optimizer = torch.optim.RMSprop(self.mlp.parameters())

        if scale_data:
            x_states = x_array[:, :state_size]
            x_actions = x_array[:, state_size:]

            self._state_scaler = StandardScaler()
            self._state_scaler.fit(x_states)
            self._action_scaler = StandardScaler()
            self._action_scaler.fit(x_actions)

            self._state_scaler.transform(x_states, copy=False)
            self._state_scaler.transform(y_array, copy=False)
            self._action_scaler.transform(x_actions, copy=False)

        n_batches = int(np.ceil(1.0 * x_array.shape[0] / batch_size))
        losses = np.zeros(epochs)

        for t in range(epochs):

            if shuffle_data:
                idx = np.random.permutation(x_array.shape[0])
                x_array = x_array[idx]
                y_array = y_array[idx]

            for batch in range(n_batches):

                idx_lb = batch_size * batch
                idx_up = min(batch_size * (batch + 1), x_array.shape[0])
                x_batch = x_array[idx_lb:idx_up]
                y_batch = y_array[idx_lb:idx_up]

                x_variable = torch.from_numpy(x_batch.astype(np.float32)).to(self.device)  # type: tensor.Tensor
                y_variable = torch.from_numpy(y_batch.astype(np.float32)).to(self.device)  # type: tensor.Tensor
                assert hasattr(y_variable, 'requires_grad')
                y_variable.requires_grad = False

                output = self.mlp.forward(x_variable)
                loss = self.loss_function(output, y_variable)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses[t] += loss.item()

            losses[t] /= x_batch.shape[0]

            if callable(epoch_callback):
                epoch_callback(t, losses[t])
            else:
                print(t, losses[t])

        return losses

    def forward_sim(self, action_sequences: np.ndarray, initial_state: np.ndarray):

        with torch.no_grad():

            n_sequences, horizon, action_size = action_sequences.shape

            if self._action_scaler is not None:
                # scale input actions
                reshaped = action_sequences.reshape(n_sequences * horizon, -1)
                self._action_scaler.transform(reshaped, copy=False)

            if self._state_scaler is not None:
                # scale initial state
                initial_state = initial_state.copy()
                self._state_scaler.transform(initial_state.reshape(1, -1), copy=False)

            state_size, = initial_state.shape
            self._check_input_sizes(state_size, action_size)
            outputs = torch.zeros((n_sequences, horizon, state_size))
            action_sequences = torch.Tensor(action_sequences).to(self.device)

            curr_states = torch.Tensor(initial_state).to(self.device)
            curr_states = curr_states.repeat((n_sequences, 1))

            for t in range(horizon):
                input_ = torch.cat([curr_states, action_sequences[:, t, :]], dim=1)
                curr_states = self.mlp.forward(input_)
                outputs[:, t, :] = curr_states

        outputs = outputs.cpu().numpy()

        if self._state_scaler is not None:
            # scale back output states
            reshaped = outputs.reshape(n_sequences * horizon, -1)
            self._state_scaler.inverse_transform(reshaped, copy=False)

        return outputs

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['mlp']
        odict['_mlp_state_dict'] = self._get_state_dict()
        return odict

    def __setstate__(self, odict):

        self.__dict__.update(odict)
        self.mlp = _MlpNetwork(self.n_inputs, self.n_outputs, self.hidden_units)
        self.mlp.load_state_dict(odict['_mlp_state_dict'])
        self.device = torch.device('cpu')

    def _get_state_dict(self):
        state_dict = self.mlp.cpu().state_dict()
        if self.device is not None and self.device.type != 'cpu':
            self.mlp.cuda()
        return state_dict

    def _check_input_sizes(self, action_size, state_size):
        if action_size + state_size != self.n_inputs:
            raise ValueError(f"Actions and state have dimension {action_size} and {state_size} respectively "
                             f"but MLP was initialized to {self.n_inputs} inputs")
