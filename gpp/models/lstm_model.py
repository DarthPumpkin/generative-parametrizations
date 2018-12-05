import math
import pickle
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, tensor

from gpp.lstm_dataset import LSTM_Dataset
from . import BaseModel


class _LstmNetwork(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_outputs, n_layers=1):
        super(_LstmNetwork, self).__init__()
        self.n_inputs, self.n_hidden, self.n_outputs, self.n_layers = n_inputs, n_hidden, n_outputs, n_layers
        self.lstm = nn.LSTM(n_inputs, n_hidden, num_layers=n_layers, batch_first=True)
        self.hidden2out = nn.Linear(n_hidden, n_outputs)

    def forward(self, x: torch.Tensor, batch_size: int=None, hidden: (torch.Tensor, torch.Tensor)=None) -> (torch.Tensor, torch.Tensor):
        if hidden is None:
            assert batch_size is not None, 'Batch size must be specified if hidden is not provided'
            hidden = self.init_hidden(batch_size, x.device)
        output, hidden = self.lstm(x, hidden)
        output = self.hidden2out(output)
        return output, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.n_layers, batch_size, self.n_hidden, device=device),
                torch.zeros(self.n_layers, batch_size, self.n_hidden, device=device))


class LSTM_Model(BaseModel):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_layers=1, np_random=None, device=torch.device("cpu")):
        """
        :param n_inputs:
        :param n_hidden: number of units in LSTM (same in each layer)
        :param n_outputs: number of outputs
        :param n_layers: number of layers within LSTM
        :param np_random:
        :param device:
        """
        super().__init__(np_random=np_random)
        self.n_inputs, self.n_hidden, self.n_outputs, self.n_layers = n_inputs, n_hidden, n_outputs, n_layers
        self.network = _LstmNetwork(n_inputs, n_hidden, n_outputs, n_layers=n_layers)

        self.loss_function = nn.MSELoss()
        self.hidden = None
        self.device = device
        self._state_scaler = None
        self._action_scaler = None
        self._target_scaler = None
        self.use_window_in_forward_sim = True

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self.network = self.network.to(value)
        self._device = value

    @staticmethod
    def load(file_path: Path, device=torch.device("cpu")):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
            if not isinstance(model, LSTM_Model):
                raise ValueError(f'Cannot load object of type {type(model)}!')
            model.device = device
            return model

    def save(self, file_path: Path):
        file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def train(self, episodes: Sequence[Tuple[np.ndarray]], targets=None, window_size=5, epochs: int = None,
              batch_size: int = None, epoch_callback=None, scale_data=False, scale_targets=False):

        if not epochs:
            raise ValueError("Missing required kwarg: epochs")

        if not batch_size:
            raise NotImplementedError()

        if targets is None and scale_data != scale_targets:
            raise ValueError

        state_size, action_size = episodes[0][0].shape[1], episodes[0][1].shape[1]
        self._check_input_sizes(action_size, state_size)
        optimizer = torch.optim.RMSprop(self.network.parameters())
        # optimizer = torch.optim.Adam(self.network.parameters(), lr=0.01)

        dataset = LSTM_Dataset(episodes, window_size, observations=targets)
        n_batches = math.ceil(len(dataset) / batch_size)
        dataset_order = np.random.permutation(len(dataset))
        # dataset_order = np.arange(len(dataset))

        n_episodes, ep_horizon = dataset.in_state.shape[:2]

        if scale_data:
            states_view = dataset.in_state.reshape(n_episodes * ep_horizon, -1)
            actions_view = dataset.actions.reshape(n_episodes * ep_horizon, -1)

            self._state_scaler = StandardScaler()
            self._state_scaler.fit(states_view)
            self._action_scaler = StandardScaler()
            self._action_scaler.fit(actions_view)

            self._state_scaler.transform(states_view, copy=False)
            self._action_scaler.transform(actions_view, copy=False)

        if scale_targets:
            out_states_view = dataset.out_state.reshape(n_episodes * ep_horizon, -1)
            self._target_scaler = StandardScaler()
            self._target_scaler.fit(out_states_view)
            self._target_scaler.transform(out_states_view, copy=False)

        losses = np.zeros(epochs)

        for t in range(epochs):

            for batch in range(n_batches):
                batch_ix = dataset_order[batch * batch_size:(batch + 1) * batch_size]
                actual_batch_size = len(batch_ix)
                batch_data = [dataset[i] for i in batch_ix]
                x_batch, a_batch, y_batch = [np.array(data) for data in zip(*batch_data)]
                input_batch = np.concatenate((x_batch, a_batch), axis=-1)

                x_variable = torch.from_numpy(input_batch.astype(np.float32)).to(self.device)  # type: tensor.Tensor
                y_variable = torch.Tensor(y_batch).to(self.device)  # type: tensor.Tensor
                assert hasattr(y_variable, 'requires_grad')
                y_variable.requires_grad = False

                output, _ = self.network.forward(x_variable, actual_batch_size)
                output = output[:, -1]

                loss = self.loss_function(output, y_variable)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses[t] += loss.item()

            losses[t] /= n_batches

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

            if self.use_window_in_forward_sim:

                window_size = min(horizon, 5)
                input_ = torch.cat([curr_states, action_sequences[:, 0, :]], dim=1)
                window = input_[:, None].repeat(1, window_size, 1)

                hidden = self.network.init_hidden(n_sequences, device=self.device)

                for t in range(horizon):
                    window[:, :window_size-1] = window[:, 1:]
                    window[:, window_size-1] = torch.cat([curr_states, action_sequences[:, t, :]], dim=1)
                    out, hidden = self.network.forward(window, hidden=hidden)
                    curr_states = out[:, -1].squeeze()
                    outputs[:, t, :] = curr_states
            else:

                hidden = self.network.init_hidden(n_sequences, device=self.device)
                for t in range(horizon):
                    input_ = torch.cat([curr_states, action_sequences[:, t, :]], dim=1)
                    out, hidden = self.network.forward(input_[:, None], hidden=hidden)
                    curr_states = out[:, -1].squeeze()
                    outputs[:, t, :] = curr_states

        outputs = outputs.cpu().numpy()

        if self._target_scaler is not None:
            # scale back output states
            reshaped = outputs.reshape(n_sequences * horizon, -1)
            self._target_scaler.inverse_transform(reshaped, copy=False)

        return outputs

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['network']
        odict['_network_state_dict'] = self._get_state_dict()
        return odict

    def __setstate__(self, odict):

        self.__dict__.update(odict)
        self.network = _LstmNetwork(self.n_inputs, self.n_hidden, self.n_outputs, n_layers=self.n_layers)
        self.network.load_state_dict(odict['_network_state_dict'])
        self.device = torch.device('cpu')

    def _get_state_dict(self):
        state_dict = self.network.cpu().state_dict()
        if self.device is not None and self.device.type != 'cpu':
            self.network.cuda()
        return state_dict

    def _check_input_sizes(self, action_size, state_size):
        if action_size + state_size != self.n_inputs:
            raise ValueError(f"Actions and state have dimension {action_size} and {state_size} respectively "
                             f"but LSTM was initialized to {self.n_inputs} inputs")
