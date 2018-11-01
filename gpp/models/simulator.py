import gym
import numpy as np
from gym.envs.classic_control import PendulumEnv

from . import BaseModel
from .utilities import get_observation_space, get_observations


class PendulumSim(BaseModel):

    def forward_sim(self, action_sequences: np.ndarray, env: gym.Env):

        raw_env = env.unwrapped # type: PendulumEnv
        if not isinstance(raw_env, PendulumEnv):
            raise ValueError('PendulumSim only works with PendulumEnv or subclasses!')

        obs_space = get_observation_space(env)
        n_sequences, horizon, _ = action_sequences.shape
        all_states = np.zeros((n_sequences, horizon+1) + obs_space.shape)

        # get initial state
        init_state = get_observations(env)
        all_states[:, 0] = np.tile(init_state, (n_sequences, 1))

        # internal states
        last_internal_states = np.tile(raw_env.state, (n_sequences, 1))

        g, m, l = raw_env.physical_props
        dt = raw_env.dt

        for t in range(horizon):

            th = last_internal_states[:, 0]
            thdot = last_internal_states[:, 1]

            u = action_sequences[:, t].squeeze()
            u = np.clip(u, -raw_env.max_torque, raw_env.max_torque)

            newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
            last_internal_states[:, 0] = th + newthdot * dt
            last_internal_states[:, 1] = np.clip(newthdot, -raw_env.max_speed, raw_env.max_speed)

            all_states[:, t+1] = np.array([
                np.cos(last_internal_states[:, 0]),
                np.sin(last_internal_states[:, 0]),
                last_internal_states[:, 1]]
            ).T

        return all_states[:, 1:]
