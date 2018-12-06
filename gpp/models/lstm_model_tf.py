import math
import pickle
from pathlib import Path
from typing import Sequence, Tuple, List

import numpy as np
import torch

from lstm import LSTMModel
from . import BaseModel


class LSTM_Model_TF(BaseModel):
    def __init__(self):
        super().__init__()

        vae_path = './tf_vae/kl2rl1-z16-b250-push_sphere_v0vae-fetch199.json'
        l2l_modelpath = './out/push_sphere_v1_l2lmodel.hdf5'
        l2reward_modelpath = './out/push_sphere_v1_l2rewardmodel.hdf5'

        self.window_size = 4
        self.model = LSTMModel(vae_path, z_size=16, steps=self.window_size, training=False, l2l_model_path=l2l_modelpath,
                               l2reward_model_path=l2reward_modelpath)

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

        start_idx = 0
        pred_z = None
        for j in range(self.window_size, horizon):
            state_action = window[:, start_idx: j] # np.concatenate([d[start_idx: j], actions[i][start_idx: j]], axis=1)
            pred_z = self.model.predict_l2l(state_action)
            start_idx += 1
            if start_idx < horizon:
                window[:, :self.window_size - 1] = window[:, 1:]
                window[:, self.window_size - 1] = np.concatenate((pred_z, action_sequences[:, start_idx]), axis=1)

        pred_rw = self.model.predict_l2reward(pred_z)
        return pred_rw
