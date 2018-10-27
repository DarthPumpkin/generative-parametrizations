import gym
import time
import numpy as np
import mujoco_py as mj
from baselines.run import make_vec_env, VecNormalize, get_env_type
from baselines.ppo2 import ppo2
from baselines.trpo_mpi import trpo_mpi
from baselines.ddpg import ddpg
from baselines.a2c import a2c
from baselines.acer import acer
from gym.envs.classic_control.pendulum import angle_normalize


def build_env(env_name, num_env=1, seed=None):
    env_type, env_id = get_env_type(env_name)

    env_ = make_vec_env(env_id, env_type, num_env, seed)

    if env_type == 'mujoco':
        env_ = VecNormalize(env_)

    return env_


if __name__ == '__main__':

    print('Loading environment...')
    env = gym.make("Pendulum-v0")
    # env = build_env("Pendulum-v0")



    #old_step = env.envs[0].env.env.step

    def step(u):

        self = env.envs[0].env.env

        th, thdot = self.state # th := theta

        g = 10.
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    # env.envs[0].env.env.step = step




    observation = env.reset()

    print('Training...')

    if False:
        rl_model = acer.learn(
            network='lstm',
            env=env,
            total_timesteps=5000
        )

    if False:
        rl_model = ppo2.learn(
            network='mlp',
            env=env,
            total_timesteps=200000
        )

    if False:
        rl_model = trpo_mpi.learn(
            network='mlp',
            env=env,
            total_timesteps=100000
        )

    if True:
        rl_model = ddpg.learn(
            network='mlp',
            env=env,
            total_timesteps=100
        )

    obs = env.reset()

    while True:
        actions, _, state, _ = rl_model.step(obs)
        obs, _, done, _ = env.step(actions)
        env.render()
        done = done.any() if isinstance(done, np.ndarray) else done

        if done:
            obs = env.reset()

    print('Done.')
