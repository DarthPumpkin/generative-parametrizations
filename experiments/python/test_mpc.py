import gym
import numpy as np
from time import sleep
from gym.envs.robotics import FetchPickAndPlaceSphereEnv

import _gpp
from gpp.mpc import MPC
from gpp.models import DummyModel, PendulumSim


if __name__ == '__main__':

    # env = gym.make('FetchPickAndPlaceDense-v1')
    env = gym.make('Pendulum-v0')
    raw_env = env.unwrapped # type: FetchPickAndPlaceSphereEnv
    raw_env.seed(42)
    np_random = raw_env.np_random

    model = PendulumSim(np_random)

    horizon = 20
    n_sequences = 2000
    controller = MPC(env, model, horizon, n_sequences, np_random)

    obs = env.reset()
    for _ in range(20000):
        env.render()
        action = controller.get_action()
        obs, rewards, dones, info = env.step(action)
        sleep(1. / 60)
