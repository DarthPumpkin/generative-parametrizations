from pathlib import Path
from time import sleep

import numpy as np
import torch
import gym
from gym.envs.classic_control import GaussianPendulumEnv

import _gpp
from gpp.mpc import MPC
from gpp.models.vision import VaeLstmTFModel, VaeTorchModel


VAE_MODEL_PATH = Path('tf_vae/kl2rl1-z6-b100-kl2rl1-b100-z6-pendulum_v0vae-fetch199.json')
#ENV_MODEL_PATH = Path('out/pendulum_v0_vision_lstm_model.pkl')

#ENV_MODEL1_PATH = Path('out/pendulum_v0_vision_mlp1_model.pkl')
#ENV_MODEL2_PATH = Path('out/pendulum_v0_vision_mlp2_model.pkl')


#l2l_model_path = Path('./out/pendulum_v0_l2lmodel.hdf5')
#l2reward_model_path = Path('./out/pendulum_v0_l2rewardmodel.hdf5')


WINDOW_SIZE = 3

if WINDOW_SIZE == 2:
    l2l_torch_path = Path('./out/pendulum_torch_models/model45-45_l2l-window2.pth')
    l2r_torch_path = Path('./out/pendulum_torch_models/model45-45_l2r-window2.pth')
elif WINDOW_SIZE == 3:
    l2l_torch_path = Path('../good_models/dynamics/pendulum/model45-45_l2l-window3.pth')
    l2r_torch_path = Path('../good_models/dynamics/pendulum/model45-45_l2r-window3.pth')
elif WINDOW_SIZE == 4:
    l2l_torch_path = Path('./out/pendulum_torch_models/model45-45_l2l-window4.pth')
    l2r_torch_path = Path('./out/pendulum_torch_models/model45-45_l2r-window4.pth')
else:
    raise NotImplementedError


VAE_ARGS = dict(z_size=6, batch_size=64)

MPC_HORIZON = 20
MPC_SEQUENCES = 32000


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

    def set_mass(new_mass, fake_mass=None):
        raw_env.sampled_mass = new_mass
        scale_range = (0.4, 1.5)
        if fake_mass is None:
            mass_for_scale = new_mass
        else:
            mass_for_scale = fake_mass
        new_scale = np.clip((mass_for_scale + 0.4) * 0.7, *scale_range)
        raw_env.length_scale = new_scale

    # model = VaeLstmModel(VAE_MODEL_PATH, ENV_MODEL_PATH, torch_device=device, **VAE_ARGS)
    # model = VaeLstmTFModel(VAE_MODEL_PATH, l2l_model_path, l2reward_model_path, window_size=3, **VAE_ARGS)
    # model = VaeMlpModel(VAE_MODEL_PATH, ENV_MODEL1_PATH, ENV_MODEL2_PATH, torch_device=device, **VAE_ARGS)
    model = VaeTorchModel(VAE_MODEL_PATH, l2l_torch_path, l2r_torch_path, torch_device=device, window_size=WINDOW_SIZE, **VAE_ARGS)
    controller = MPC(env, model, MPC_HORIZON, MPC_SEQUENCES, np_random, use_history=True, direct_reward=True)

    model.z_filter = [2, 5]

    set_mass(0.2, 1.4)

    for e in range(2000):
        env.reset()
        controller.forget_history()
        for s in range(300):
            rgb_obs = env.render(mode='rgb_array')
            action = controller.get_action(rgb_obs)
            _, rewards, dones, info = env.step(action)
            print(rewards)
            sleep(1. / 60)


if __name__ == '__main__':
    main()
