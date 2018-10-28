import gym
import numpy as np

from . import BaseModel
from .utilities import get_observation_space


class DummyModel(BaseModel):

    def forward_sim(self, action_sequences: np.ndarray, env: gym.Env):
        obs_space = get_observation_space(env)
        lb = np.finfo(np.float32).min
        ub = np.finfo(np.float32).max
        low = obs_space.low.clip(lb, ub)
        high = obs_space.high.clip(lb, ub)
        shape = action_sequences.shape[:2] + (low.size,)
        resulting_states = self.np_random.uniform(low, high, shape)
        return resulting_states
