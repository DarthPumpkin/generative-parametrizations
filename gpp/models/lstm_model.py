import math
import pickle
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from torch import nn, tensor
from sklearn.preprocessing import StandardScaler

from gpp.models.utilities import merge_episodes
from . import BaseModel
from ..mdn import MDN, sample_gmm_torch, mdn_loss


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
        self.lstm = nn.LSTM(n_inputs, n_hidden, num_layers=n_layers, batch_first=True)
        self.hidden2out = nn.Linear(n_hidden, n_outputs)
        self.loss_function = nn.MSELoss()
        self.hidden = None
        self.device = device
        self._state_scaler = None
        self._action_scaler = None

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self.lstm = self.lstm.to(value)
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

    def train(self, episodes: Sequence[Tuple[np.ndarray]], epochs: int=None, batch_size: int=None, epoch_callback=None,
              scale_data=False):

        if not epochs:
            raise ValueError("Missing required kwarg: epochs")

        if not batch_size:
            raise NotImplementedError()

        state_size, action_size = episodes[0][0].shape[1], episodes[0][1].shape[1]
        self._check_input_sizes(action_size, state_size)
        optimizer = torch.optim.RMSprop(self.lstm.parameters())

        if scale_data:
            x_array, y_array = merge_episodes(episodes)
            x_states = x_array[:, :state_size]
            x_actions = x_array[:, state_size:]

            self._state_scaler = StandardScaler()
            self._state_scaler.fit(x_states)
            self._action_scaler = StandardScaler()
            self._action_scaler.fit(x_actions)

            for episode in episodes:
                states, actions = episode
                self._state_scaler.transform(states, copy=False)
                self._action_scaler.transform(actions, copy=False)

        n_batches = int(math.ceil(len(episodes) / batch_size))
        losses = np.zeros(epochs)

        for t in range(epochs):

            for batch in range(n_batches):
                batch_data = episodes[batch * batch_size:(batch + 1) * batch_size]
                x_batch, y_batch, a_batch = zip(*[(states[:-1], states[1:], actions) for (states, actions) in
                                                  batch_data])
                actual_batch_size = len(x_batch)
                input_batch = np.concatenate((x_batch, a_batch), axis=-1)

                x_variable = torch.from_numpy(x_batch.astype(np.float32)).to(self.device)  # type: tensor.Tensor
                y_variable = torch.from_numpy(y_batch.astype(np.float32)).to(self.device)  # type: tensor.Tensor
                assert hasattr(y_variable, 'requires_grad')
                y_variable.requires_grad = False

                initial_hidden = self._init_lstm(batch_size)
                output, hidden = self.lstm(x_variable, initial_hidden)
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
                pi, mu, sig2 = self.mdn.forward(input_)
                curr_states = sample_gmm_torch(pi, mu, sig2)
                outputs[:, t, :] = curr_states

        outputs = outputs.cpu().numpy()

        if self._state_scaler is not None:
            # scale back output states
            reshaped = outputs.reshape(n_sequences * horizon, -1)
            self._state_scaler.inverse_transform(reshaped, copy=False)

        return outputs

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['mdn']
        odict['_mdn_state_dict'] = self._get_state_dict()
        return odict

    def __setstate__(self, odict):

        # for compatibility with older models
        if 'device' in odict.keys():
            del odict['device']
        if '_action_scaler' not in odict.keys():
            odict['_action_scaler'] = None
            odict['_state_scaler'] = None

        self.__dict__.update(odict)
        self.mdn = MDN(self.n_inputs, self.n_outputs, self.n_components)
        self.mdn.load_state_dict(odict['_mdn_state_dict'])
        self.device = torch.device('cpu')

    def _get_state_dict(self):
        state_dict = self.mdn.cpu().state_dict()
        if self.device is not None and self.device.type != 'cpu':
            self.mdn.cuda()
        return state_dict

    def _check_input_sizes(self, action_size, state_size):
        if action_size + state_size != self.mdn.n_inputs:
            raise ValueError(f"Actions and state have dimension {action_size} and {state_size} respectively "
                             f"but MDN was initialized to {self.mdn.n_inputs} inputs")

    def _init_lstm(self, batch_size):
        return (torch.zeros(batch_size, self.n_layers, self.n_hidden),
                torch.zeros(batch_size, self.n_layers, self.n_hidden))
