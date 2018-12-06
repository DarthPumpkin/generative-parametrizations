from pathlib import Path
from time import sleep

import torch
import gym
from gym.envs.classic_control import GaussianPendulumEnv

import _gpp
from gpp.mpc import MPC
from gpp.models import VaeLstmModel


VAE_MODEL_PATH = Path('')
ENV_MODEL_PATH = Path('')
VAE_ARGS = dict(z_size=16, batch_size=32)

MPC_HORIZON = 5
MPC_SEQUENCES = 2000


def main():

    if torch.cuda.is_available():
        print("CUDA available, proceeding with GPU...")
        device = torch.device("cuda")
    else:
        print("No GPU found, proceeding with CPU...")
        device = torch.device("cpu")

    env = gym.make('GaussianPendulum-v0')

    raw_env = env.unwrapped # type: GaussianPendulumEnv
    raw_env.seed(42)
    np_random = raw_env.np_random

    model = VaeLstmModel(VAE_MODEL_PATH, ENV_MODEL_PATH, torch_device=device, **VAE_ARGS)
    controller = MPC(env, model, MPC_HORIZON, MPC_SEQUENCES, np_random)

    for e in range(2000):
        env.reset()
        for s in range(100):
            rgb_obs = env.render(mode='rgb_array')
            action = controller.get_action(rgb_obs)
            _, rewards, dones, info = env.step(action)
            sleep(1. / 60)


if __name__ == '__main__':
    main()
