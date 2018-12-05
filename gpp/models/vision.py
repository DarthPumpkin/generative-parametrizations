from pathlib import Path

import torch
import numpy as np

from gpp.world_models_vae import ConvVAE
from . import BaseModel, LSTM_Model


class VisionModel(BaseModel):

    def __init__(self, vae_model_path: Path, env_model: BaseModel, z_size: int=16, batch_size: int=32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vae = ConvVAE(z_size=z_size, batch_size=batch_size, is_training=False, reuse=False, gpu_mode=False)
        vae.load_json(vae_model_path)
        self.vae = vae
        self.vae_input_shape = (64, 64, 3)
        self.env_model = env_model

    def forward_sim(self, action_sequences: np.ndarray, initial_state: np.ndarray):
        assert initial_state.shape == self.vae_input_shape, 'Initial state must be an image!'

        if initial_state.max() > 1.0:
            initial_state /= 255.

        dummy_batch = np.zeros((self.vae.batch_size,) + self.vae_input_shape)
        dummy_batch[0] = initial_state
        initial_z = self.vae.encode(dummy_batch)[0]

        return self.env_model.forward_sim(action_sequences, initial_z)


class VaeLstmModel(VisionModel):

    def __init__(self, vae_model_path: Path, lstm_model_path: Path, torch_device=None, *args, **kwargs):
        torch_device = torch_device or torch.device('cpu')
        lstm = LSTM_Model.load(lstm_model_path, torch_device)
        super().__init__(vae_model_path=vae_model_path, world_model=lstm, *args, **kwargs)
