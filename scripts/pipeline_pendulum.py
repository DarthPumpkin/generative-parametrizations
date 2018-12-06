from pathlib import Path
from time import sleep

import numpy as np
import torch
import gym
from gym.envs.classic_control import GaussianPendulumEnv

import _gpp
from gpp.mpc import MPC
from gpp.models import VaeLstmModel
from gpp.models.vision import VaeMlpModel


VAE_MODEL_PATH = Path('tf_vae/kl2rl1-z6-b100-kl2rl1-b100-z6-pendulum_v0vae-fetch199.json')
ENV_MODEL_PATH = Path('out/pendulum_v0_vision_lstm_model.pkl')

ENV_MODEL1_PATH = Path('out/pendulum_v0_vision_mlp1_model.pkl')
ENV_MODEL2_PATH = Path('out/pendulum_v0_vision_mlp2_model.pkl')

VAE_ARGS = dict(z_size=6, batch_size=64)

MPC_HORIZON = 20
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

    def set_mass(new_mass):
        raw_env.sampled_mass = new_mass
        scale_range = (0.4, 1.5)
        new_scale = np.clip((new_mass + 0.4) * 0.7, *scale_range)
        raw_env.length_scale = new_scale

    # model = VaeLstmModel(VAE_MODEL_PATH, ENV_MODEL_PATH, torch_device=device, **VAE_ARGS)
    model = VaeMlpModel(VAE_MODEL_PATH, ENV_MODEL1_PATH, ENV_MODEL2_PATH, torch_device=device, **VAE_ARGS)
    controller = MPC(env, model, MPC_HORIZON, MPC_SEQUENCES, np_random)

    set_mass(0.2)

    for e in range(2000):
        env.reset()
        for s in range(100):
            rgb_obs = env.render(mode='rgb_array')
            action = controller.get_action(rgb_obs)
            _, rewards, dones, info = env.step(action)
            sleep(1. / 60)


if __name__ == '__main__':
    main()
