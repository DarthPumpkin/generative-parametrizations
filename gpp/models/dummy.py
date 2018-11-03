import gym
import numpy as np

from . import BaseModel
from .utilities import get_observation_space


class DummyModel(BaseModel):

    def __init__(self, env: gym.Env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_space = get_observation_space(env)

    def forward_sim(self, action_sequences: np.ndarray, initial_state: np.ndarray):
        lb = np.finfo(np.float32).min
        ub = np.finfo(np.float32).max
        low = self.obs_space.low.clip(lb, ub)
        high = self.obs_space.high.clip(lb, ub)
        shape = action_sequences.shape[:2] + (low.size,)
        resulting_states = self.np_random.uniform(low, high, shape)
        return resulting_states
