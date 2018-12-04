import copy

from joblib import Parallel, delayed
import gym
import numpy as np
from gym.envs.classic_control import PendulumEnv, CartPoleEnv

try:
    from gym.envs.robotics import FetchEnv
except (ImportError, gym.error.DependencyNotInstalled) as e:
    FetchEnv = None
    print('WARNING: Could not import robotics environments! MuJoCo might not be installed.')


from . import BaseModel
from .utilities import get_observations


class PendulumSim(BaseModel):

    def __init__(self, env: gym.Env, *args, **kwargs):
        super().__init__(*args, **kwargs)

        raw_env = env.unwrapped  # type: PendulumEnv
        if not isinstance(raw_env, PendulumEnv):
            raise ValueError('PendulumSim only works with PendulumEnv or subclasses!')

        self.env = env
        self.raw_env = raw_env

    def forward_sim(self, action_sequences: np.ndarray, initial_state: np.ndarray):

        n_sequences, horizon, _ = action_sequences.shape
        all_states = np.zeros((n_sequences, horizon+1) + initial_state.shape)
        raw_env = self.raw_env

        # get initial state
        all_states[:, 0] = np.tile(initial_state, (n_sequences, 1))

        # internal states
        last_internal_states = np.tile(raw_env.state, (n_sequences, 1))

        extended_state = None
        if initial_state.shape[0] > 3:
            extended_state = initial_state[3:]
            extended_state = np.tile(extended_state.reshape(1, -1), (n_sequences, 1))

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

            new_state = np.array([
                np.cos(last_internal_states[:, 0]),
                np.sin(last_internal_states[:, 0]),
                last_internal_states[:, 1]]
            ).T

            if extended_state is not None:
                new_state = np.concatenate([new_state, extended_state], axis=1)

            all_states[:, t+1] = new_state

        return all_states[:, 1:]


class CartPoleSim(BaseModel):

    def __init__(self, env: gym.Env, *args, **kwargs):
        super().__init__(*args, **kwargs)

        raw_env = env.unwrapped  # type: CartPoleEnv
        if not isinstance(raw_env, CartPoleEnv):
            raise ValueError('CartPoleSim only works with CartPoleEnv or subclasses!')

        self.env = env
        self.raw_env = raw_env

    def forward_sim(self, action_sequences: np.ndarray, initial_state: np.ndarray):

        n_sequences, horizon = action_sequences.shape
        all_states = np.zeros((n_sequences, horizon+1) + initial_state.shape)
        raw_env = self.raw_env

        # get initial state
        all_states[:, 0] = np.tile(initial_state, (n_sequences, 1))

        # internal states
        last_internal_states = np.tile(raw_env.state, (n_sequences, 1))

        force_mag = raw_env.force_mag
        polemass_length = raw_env.polemass_length
        pole_length = raw_env.length
        total_mass = raw_env.total_mass
        masspole = raw_env.masspole
        kinematics_integrator = raw_env.kinematics_integrator
        tau = raw_env.tau
        gravity = raw_env.gravity
        force = np.zeros(n_sequences)

        for t in range(horizon):

            x = last_internal_states[:, 0]
            x_dot = last_internal_states[:, 1]
            theta = last_internal_states[:, 2]
            theta_dot = last_internal_states[:, 3]

            u = action_sequences[:, t].squeeze()
            force[:] = force_mag
            force[u != 1] = -force_mag

            costheta = np.cos(theta)
            sintheta = np.sin(theta)
            temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
            thetaacc = (gravity * sintheta - costheta * temp) / (
                    pole_length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass))
            xacc = temp - polemass_length * thetaacc * costheta / total_mass

            if kinematics_integrator == 'euler':
                x = x + tau * x_dot
                x_dot = x_dot + tau * xacc
                theta = theta + tau * theta_dot
                theta_dot = theta_dot + tau * thetaacc
            else:  # semi-implicit euler
                x_dot = x_dot + tau * xacc
                x = x + tau * x_dot
                theta_dot = theta_dot + tau * thetaacc
                theta = theta + tau * theta_dot

            last_internal_states[:, 0] = x
            last_internal_states[:, 1] = x_dot
            last_internal_states[:, 2] = theta
            last_internal_states[:, 3] = theta_dot

            all_states[:, t+1] = last_internal_states.copy()

        return all_states[:, 1:]


class RoboticsSim(BaseModel):

    def __init__(self, env: gym.Env, workers: int=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if FetchEnv is None:
            raise ImportError

        raw_env = env.unwrapped  # type: FetchEnv
        if not isinstance(raw_env, FetchEnv):
            raise ValueError('PendulumSim only works with PendulumEnv or subclasses!')

        self._main_env = env
        self._workers = workers
        self._cloned_envs = [copy.deepcopy(env) for _ in range(self._workers)]

    @classmethod
    def _forward_sim_worker(cls, env: gym.Env, initial_sim_state: object,
                            action_sequences: np.ndarray, initial_state: np.ndarray):

        n_sequences, horizon, _ = action_sequences.shape
        all_states = np.zeros((n_sequences, horizon + 1) + initial_state.shape)
        raw_env = env.unwrapped  # type: FetchEnv
        initial_sim_state = copy.deepcopy(initial_sim_state)

        # get initial state
        all_states[:, 0] = np.tile(initial_state, (n_sequences, 1))

        for s in range(n_sequences):
            env.reset()
            raw_env.sim.set_state(initial_sim_state)
            raw_env.sim.forward()
            for t in range(horizon):
                a = action_sequences[s, t]
                env.step(a)
                obs = get_observations(env)
                all_states[:, t + 1] = obs.copy()

        return all_states[:, 1:]

    def _parallel_forward_sim(self, initial_sim_state: object, action_sequences: np.ndarray, initial_state: np.ndarray):

        n_sequences, horizon, _ = action_sequences.shape

        workers = min(self._workers, n_sequences)
        parallel = Parallel(n_jobs=workers, backend='multiprocessing', verbose=5)
        sequences_idx = np.array_split(np.arange(n_sequences), workers)

        results = parallel(delayed(RoboticsSim._forward_sim_worker)(
            self._cloned_envs[i],
            initial_sim_state,
            action_sequences[sequences_idx[i]],
            initial_state.copy()
        ) for i in range(workers))

        all_states = np.concatenate(results)
        return all_states


    def _fast_sim(self, action_sequences: np.ndarray, initial_state: np.ndarray):
        pass


    def forward_sim(self, action_sequences: np.ndarray, initial_state: np.ndarray):

        main_raw_env = self._main_env.unwrapped  # type: FetchEnv
        initial_sim_state = main_raw_env.sim.get_state()

        if self._workers > 1:
            return self._parallel_forward_sim(initial_sim_state, action_sequences, initial_state)
        else:
            env = self._cloned_envs[0]
            return self._forward_sim_worker(env, initial_sim_state, action_sequences, initial_state)
