import gym
import time
import numpy as np
import mujoco_py as mj
from baselines.ppo2 import ppo2
from baselines.trpo_mpi import trpo_mpi
from baselines.ddpg import ddpg

if __name__ == '__main__':

    print('Loading environment...')
    env = gym.make("FetchPickAndPlace-v1")
    #env = gym.make("Ant-v2")

    env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])



    old_step = env.step

    def step(*args, **kwargs):
        res = old_step(*args, **kwargs)
        env.env.env.sim.model.opt.gravity[2] = np.random.random()
        return res

    env.step = step




    observation = env.reset()

    print('Training...')

    if False:
        ppo_model = ppo2.learn(
            network='lstm',
            env=env.unwrapped,
            total_timesteps=1000
        )

    if False:
        trpo_model = trpo_mpi.learn(
            network='lstm',
            env=env.unwrapped,
            total_timesteps=1000
        )

    if True:
        ddpg_model = ddpg.learn(
            network='mlp',
            env=env,
            total_timesteps=1000
        )

        obs = env.reset()

        while True:
            actions, _, state, _ = ddpg_model.step(obs)
            obs, _, done, _ = env.step(actions)
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done

            if done:
                obs = env.reset()

    print('Done.')
