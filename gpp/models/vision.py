from pathlib import Path

from cv2 import resize, INTER_AREA
import torch
import numpy as np

from gpp.world_models_vae import ConvVAE
from . import BaseModel, LSTM_Model
from .mlp_model import ComboMlpModel
from .torch_model import ComboTorchModel
from .lstm_model_tf import LSTM_Model_TF


class VisionModel(BaseModel):

    def __init__(self, vae_model_path: Path, env_model: BaseModel=None, z_filter=None, z_size: int=16, batch_size: int=32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vae = ConvVAE(z_size=z_size, batch_size=batch_size, is_training=False, reuse=False, gpu_mode=False)
        vae.load_json(vae_model_path)
        self.vae = vae
        self.vae_input_shape = (64, 64, 3)
        self.env_model = env_model
        self.z_filter = z_filter

    def vae_decode(self, latent_states: np.ndarray):

        if len(latent_states.shape) == 3:
            latent_states = latent_states[None]

        n_states = len(latent_states)
        if n_states > self.vae.batch_size:
            raise NotImplementedError

        z_filter = np.arange(self.vae.z_size)
        if self.z_filter is not None:
            z_filter = self.z_filter

        dummy_batch = np.zeros((self.vae.batch_size, self.vae.z_size), dtype=np.float32)
        dummy_batch[:n_states, z_filter] = latent_states

        decoded = self.vae.decode(dummy_batch)[:n_states]
        return decoded

    def forward_sim(self, action_sequences: np.ndarray, initial_state: np.ndarray, encoding_only=False, **kwargs):
        assert len(initial_state.shape) == 3, 'Initial state must be an image!'

        initial_state = resize(initial_state, dsize=(64, 64), interpolation=INTER_AREA)

        if initial_state.max() > 1.0:
            initial_state = initial_state.astype(np.float32)
            initial_state /= 255.

        dummy_batch = np.zeros((self.vae.batch_size,) + self.vae_input_shape, dtype=np.float32)
        dummy_batch[0] = initial_state
        initial_z = self.vae.encode(dummy_batch)[0]

        if self.z_filter is not None:
            initial_z = initial_z[self.z_filter]

        if encoding_only:
            return initial_z

        sim_result = self.env_model.forward_sim(action_sequences, initial_z, **kwargs)
        return initial_z, sim_result


class VaeLstmModel(VisionModel):

    def __init__(self, vae_model_path: Path, lstm_model_path: Path, torch_device=None, *args, **kwargs):
        torch_device = torch_device or torch.device('cpu')
        lstm = LSTM_Model.load(lstm_model_path, torch_device)
        super().__init__(vae_model_path=vae_model_path, env_model=lstm, *args, **kwargs)


class VaeLstmTFModel(VisionModel):

    def __init__(self, vae_model_path: Path, l2l_model_path: Path, l2reward_model_path: Path, window_size, *args, **kwargs):
        super().__init__(vae_model_path=vae_model_path, env_model=None, *args, **kwargs)
        lstm = LSTM_Model_TF(l2l_model_path=l2l_model_path, l2reward_model_path=l2reward_model_path, window_size=window_size)
        self.env_model = lstm


class VaeMlpModel(VisionModel):

    def __init__(self, vae_model_path: Path, mlp1_path: Path, mlp2_path: Path, torch_device=None, *args, **kwargs):
        torch_device = torch_device or torch.device('cpu')
        combo_mlps = ComboMlpModel(mlp1_path, mlp2_path, device=torch_device)
        super().__init__(vae_model_path=vae_model_path, env_model=combo_mlps, *args, **kwargs)


class VaeTorchModel(VisionModel):

    def __init__(self, vae_model_path: Path, l2l_model_path: Path, l2r_model_path: Path, torch_device=None,
                 window_size=1, z_filter=None, *args, **kwargs):
        torch_device = torch_device or torch.device('cpu')
        combo_torch_model = ComboTorchModel(l2l_model_path, l2r_model_path, device=torch_device, window_size=window_size)
        super().__init__(vae_model_path=vae_model_path, env_model=combo_torch_model, z_filter=z_filter, *args, **kwargs)
