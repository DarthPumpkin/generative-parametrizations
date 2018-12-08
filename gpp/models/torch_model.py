import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from . import BaseModel


class ComboTorchModel(BaseModel):

    def __init__(self, l2l_model_path, l2r_model_path, window_size=1, device=torch.device("cpu")):
        super().__init__()
        self.l2l = torch.load(l2l_model_path).to(device)
        self.l2r = torch.load(l2r_model_path).to(device)
        self._device = device

        self._z_scaler = None
        self._a_scaler = None
        self._r_scaler = None
        self.window_size = window_size

        if hasattr(self.l2l, 'z_scaler'):
            self._z_scaler = self.l2l.z_scaler # type: StandardScaler
        if hasattr(self.l2l, 'a_scaler'):
            self._a_scaler = self.l2l.a_scaler # type: StandardScaler
        if hasattr(self.l2r, 'r_scaler'):
            self._r_scaler = self.l2r.r_scaler # type: StandardScaler

    def forward_sim(self, action_sequences: np.ndarray, initial_state: np.ndarray, **kwargs):

        if self.window_size > 1:
            return self.forward_sim_windowed(action_sequences, initial_state, **kwargs)

        with torch.no_grad():
            n_sequences, horizon, action_size = action_sequences.shape
            z_size = initial_state.shape[0]

            if self._a_scaler is not None:
                # scale input actions
                reshaped = action_sequences.reshape(n_sequences * horizon, -1)
                self._a_scaler.transform(reshaped, copy=False)

            if self._z_scaler is not None:
                # scale initial state
                initial_state = initial_state.copy()
                self._z_scaler.transform(initial_state.reshape(1, -1), copy=False)

            all_z = torch.zeros((n_sequences, horizon, z_size))
            rewards = torch.zeros(n_sequences)
            action_sequences = torch.Tensor(action_sequences).to(self._device)

            curr_z = torch.Tensor(initial_state).to(self._device)
            curr_z = curr_z.repeat((n_sequences, 1))

            for t in range(horizon):
                input_z_and_a = torch.cat([curr_z, action_sequences[:, t, :]], dim=1)
                curr_z = self.l2l.forward(input_z_and_a)
                rew = self.l2r.forward(curr_z)

                if self._r_scaler is not None:
                    self._r_scaler.inverse_transform(rew, copy=False)

                rewards += rew
                all_z[:, t, :] = curr_z

            return rewards.cpu().numpy()

    def forward_sim_windowed(self, action_sequences: np.ndarray, initial_state: np.ndarray, history=None,
                             flatten_window=True, **kwargs):

        with torch.no_grad():
            n_sequences, horizon, action_size = action_sequences.shape
            z_size = initial_state.shape[0]

            if self._a_scaler is not None:
                # scale input actions
                reshaped = action_sequences.reshape(n_sequences * horizon, -1)
                self._a_scaler.transform(reshaped, copy=False)

            if self._z_scaler is not None:
                # scale initial state
                initial_state = initial_state.copy()
                self._z_scaler.transform(initial_state.reshape(1, -1), copy=False)

            window0 = np.concatenate((initial_state, np.zeros(action_size)))
            window = np.tile(window0, (n_sequences, self.window_size, 1))

            if history:
                s_history, a_history = history
                j = 0
                for i in range(1, self.window_size):
                    if j >= len(s_history):
                        break
                    past_s, past_a = s_history[-i].copy(), a_history[-i].copy()

                    if self._a_scaler is not None:
                        self._a_scaler.transform(past_a.reshape(1, -1), copy=False)
                    if self._z_scaler is not None:
                        self._z_scaler.transform(past_s.reshape(1, -1), copy=False)

                    foo = np.concatenate((past_s, past_a))
                    foo = np.tile(foo, (n_sequences, 1))
                    window[:, self.window_size-i-1, :] = foo
                    j += 1

            window[:, self.window_size - 1, z_size:] = action_sequences[:, 0]

            rewards = np.zeros(n_sequences)
            action_sequences = torch.Tensor(action_sequences).to(self._device)
            window = torch.Tensor(window).to(self._device)

            # extend action_sequences with dummy action sequence for t = horizon-1
            dummy_as = torch.zeros((n_sequences, 1, 1)).to(self._device)
            action_sequences = torch.cat((action_sequences, dummy_as), dim=1)

            w_len = window.view(n_sequences, -1).shape[1]
            flat_window_idx_no_action = [i for i in range(w_len) if i % (self.window_size + action_size) not in range(z_size, z_size + action_size)]
            flat_window_idx = flat_window_idx_no_action + [*range(w_len - action_size, w_len)]

            for t in range(0, horizon):

                if flatten_window:
                    l2l_window = window.view(n_sequences, -1)[:, flat_window_idx]
                else:
                    l2l_window = window

                curr_z = self.l2l.forward(l2l_window)

                window[:, :self.window_size - 1] = window[:, 1:]
                window[:, self.window_size - 1] = torch.cat((curr_z, action_sequences[:, t+1]), dim=1)

                if flatten_window:
                    l2r_window = window.view(n_sequences, -1)[:, flat_window_idx_no_action]
                else:
                    l2r_window = window[:, :, -action_size:]

                rew = self.l2r.forward(l2r_window)

                rew_np = rew.cpu().numpy()
                if self._r_scaler is not None:
                    self._r_scaler.inverse_transform(rew_np, copy=False)
                rewards += rew_np.squeeze()

            return rewards
