import gym
import numpy as np
from gym.envs.robotics import FetchPickAndPlaceSphereEnv

import _gpp
from gpp.mpc import MPC
from gpp.models import BaseModel


if __name__ == '__main__':

    model = BaseModel()

    env = gym.make('FetchPickAndPlaceSphere-v1')
    raw_env = env.unwrapped # type: FetchPickAndPlaceSphereEnv
    raw_env.seed(42)
    np_random = raw_env.np_random

    horizon = 10
    n_sequences = 100
    controller = MPC(env, model, horizon, n_sequences, np_random)

    print(controller.get_action())
