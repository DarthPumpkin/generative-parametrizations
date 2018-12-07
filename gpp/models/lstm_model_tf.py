import math
import pickle
from pathlib import Path
from typing import Sequence, Tuple, List

import numpy as np
import torch

from lstm import LSTMModel
from . import BaseModel


class LSTM_Model_TF(BaseModel):

    def __init__(self, l2l_model_path: Path, l2reward_model_path: Path, window_size=4):
        super().__init__()

        self.window_size = window_size
        self.model = LSTMModel(steps=self.window_size, training=False, l2l_model_path=l2l_model_path.as_posix(),
                               l2reward_model_path=l2reward_model_path.as_posix())

    @staticmethod
    def load(file_path: Path, device=torch.device("cpu")):
        raise NotImplementedError

    def forward_sim(self, action_sequences: np.ndarray, initial_state: np.ndarray,
                    history: Tuple[List[np.array]] = None, **kwargs):

        state_size, = initial_state.shape
        n_sequences, horizon, action_size = action_sequences.shape

        window0 = np.concatenate((initial_state, np.zeros(action_size)))
        window = np.tile(window0, (n_sequences, self.window_size, 1))

        if history:
            s_history, a_history = history
            j = 0
            for i in range(1, self.window_size):
                if j >= len(s_history):
                    break
                foo = np.concatenate((s_history[-i], a_history[-i]))
                foo = np.tile(foo, (n_sequences, 1))
                window[:, self.window_size-i-1, :] = foo
                j += 1

        window[:, self.window_size - 1, state_size:] = action_sequences[:, 0]
        output = np.zeros((n_sequences, horizon))

        start_idx = 0
        for j in range(horizon):

            pred_z = self.model.predict_l2l(window)

            pred_rw = self.model.predict_l2reward(pred_z)
            output[:, start_idx] = pred_rw.squeeze()

            start_idx += 1
            if start_idx < horizon:
                window[:, :self.window_size - 1] = window[:, 1:]
                window[:, self.window_size - 1] = np.concatenate((pred_z, action_sequences[:, start_idx]), axis=1)

        return output
