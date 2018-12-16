from pathlib import Path
from time import sleep

import imageio
import cv2
import pandas as pd
from tqdm import tqdm
import itertools as it
import numpy as np
import torch
import gym
from gym.envs.classic_control import GaussianPendulumEnv

import _gpp
from gpp.mpc import MPC
from gpp.mpc_by_sgd import MpcBySgd
from gpp.models.vision import VaeTorchModel
from pendulum_evaluate import TEST_MASS_MEAN

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

Z_FILTER = [2, 5]
VAE_ARGS = dict(z_size=6, batch_size=64)

MPC_HORIZON = 8
MPC_SEQUENCES = 32000
SAVE_GIFS = True


def set_mass(raw_env, new_mass, fake_mass=None):
    raw_env.sampled_mass = new_mass
    scale_range = (0.4, 1.5)
    if fake_mass is None:
        mass_for_scale = new_mass
    else:
        mass_for_scale = fake_mass
    new_scale = np.clip((mass_for_scale + 0.4) * 0.7, *scale_range)
    raw_env.length_scale = new_scale


def init():

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

    model = VaeTorchModel(VAE_MODEL_PATH, l2l_torch_path, l2r_torch_path, torch_device=device,
                          window_size=WINDOW_SIZE, z_filter=Z_FILTER, **VAE_ARGS)
    controller = MPC(env, model, MPC_HORIZON, MPC_SEQUENCES, np_random, use_history=True, direct_reward=True)

    return env, model, controller


def visual_test():

    env, model, controller = init()
    raw_env = env.unwrapped

    set_mass(raw_env, 0.250, 1.3)

    for e in range(2000):
        env.reset()
        controller.forget_history()
        for s in range(100):
            rgb_obs = env.render(mode='rgb_array')
            action = controller.get_action(rgb_obs)
            _, rewards, dones, info = env.step(action)
            print(rewards)
            sleep(1. / 60)


def performance_test(seed=42, output_path=None, overwrite_data=False):

    prev_df = None
    if Path(output_path).exists() and not overwrite_data:
        prev_df = pd.read_pickle(output_path)

    env, model, controller = init()
    raw_env = env.unwrapped

    repetitions = 100
    ep_length = 200
    n_masses = 5

    tot_iters = repetitions * n_masses * n_masses
    masses = np.linspace(*TEST_MASS_MEAN, n_masses)
    results = []
    results_df = None

    for real_mass, visual_mass, rep in tqdm(it.product(masses, masses, range(repetitions)), total=tot_iters):

        if prev_df[(prev_df['real_mass'] == real_mass) &
                   (prev_df['visual_mass'] == visual_mass) &
                   (prev_df['repetition'] == rep)].size > 0:
            print(f'Config. rm {real_mass}, vm {visual_mass}, rep {rep} found. Skipping...')
            continue

        env.seed(seed + rep)
        controller.np_random = raw_env.np_random
        env.reset()
        controller.forget_history()

        set_mass(raw_env, real_mass, visual_mass)

        for s in range(ep_length):
            rgb_obs = env.render(mode='rgb_array')
            action = controller.get_action(rgb_obs)
            reward = env.step(action)[1]
            results.append(dict(
                repetition=rep,
                step=s,
                reward=reward,
                real_mass=real_mass,
                visual_mass=visual_mass,
                pendulum_length=raw_env.length_scale
            ))

        results_df = pd.DataFrame(results)
        if prev_df is not None:
            results_df = pd.concat((prev_df, results_df))
        if output_path is not None:
            results_df.to_pickle(output_path)

    return results_df


def demo(dream_len=20):

    dream_len = min(dream_len, MPC_HORIZON)

    env, model, controller = init()
    raw_env = env.unwrapped

    set_mass(raw_env, 0.350, 0.800)
    interrupt = False

    max_episodes = 5 if SAVE_GIFS else 20000
    episode_length = 50 if SAVE_GIFS else 200

    for e in range(max_episodes):
        env.reset()
        controller.forget_history()
        frames = []
        for s in range(episode_length):
            rgb_obs = env.render(mode='rgb_array')
            action, dream = controller.get_action(rgb_obs, return_dream=True)

            dreamed_imgs = model.vae_decode(dream[:dream_len])
            for i, img in enumerate(dreamed_imgs):
                img = cv2.resize(img, dsize=(500, 500), interpolation=cv2.INTER_LINEAR)

                if SAVE_GIFS:
                    comp_img = np.zeros((500, 500 * 2, 3), dtype=np.float32)
                    comp_img[:, :500] = rgb_obs / 255.
                    comp_img[:, 500:] = img
                    comp_img = (comp_img * 255).astype(np.uint8)
                    frames.append(comp_img)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow('frame', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    interrupt = True
                    break
                sleep(0.02)

            reward = env.step(action)[1]
            print('Reward: ', reward)

            if interrupt:
                break

        if SAVE_GIFS:
            gif_path = f'./presentation_pendulum_dream_{e}.gif'
            imageio.mimsave(gif_path, frames, fps=29)

        if interrupt:
            break

    cv2.destroyAllWindows()


def vae_real_vs_decoded():

    env, model, controller = init()
    raw_env = env.unwrapped

    set_mass(raw_env, 0.350, 0.800)
    interrupt = False

    for e in range(5):
        env.reset()
        controller.forget_history()
        frames = []
        for s in range(200):
            rgb_obs = env.render(mode='rgb_array')
            action = controller.get_action(rgb_obs)

            vae_sight = model.vae_decode(model.vae_encode(rgb_obs), apply_filter=False)[0]
            vae_sight = cv2.resize(vae_sight, dsize=(500, 500), interpolation=cv2.INTER_LINEAR)

            comp_img = np.zeros((500, 500*2, 3), dtype=np.float32)
            comp_img[:, :500] = rgb_obs / 255.
            comp_img[:, 500:] = vae_sight
            comp_img = (comp_img * 255).astype(np.uint8)
            frames.append(comp_img)

            vae_sight = cv2.cvtColor(comp_img, cv2.COLOR_RGB2BGR)
            cv2.imshow('frame', vae_sight)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                interrupt = True
                break

            env.step(action)
            if interrupt:
                break

        if SAVE_GIFS:
            gif_path = f'./presentation_pendulum_vae_{e}.gif'
            imageio.mimsave(gif_path, frames, fps=29)

        if interrupt:
            break

    cv2.destroyAllWindows()


def visual_test_mpc_by_sgd():

    env, model, controller = init()
    raw_env = env.unwrapped

    controller = MpcBySgd(env, model, MPC_HORIZON, MPC_SEQUENCES, device=torch.device('cuda'))

    set_mass(raw_env, 1.0, 1.0)

    for e in range(10):
        env.reset()
        controller.forget_history()
        frames = []

        for _ in tqdm(range(200), desc='STEP'):
            rgb_obs = env.render(mode='rgb_array')

            if SAVE_GIFS:
                frames.append(rgb_obs.copy())

            action = controller.get_action(rgb_obs)
            _, rewards, dones, info = env.step(action)
            print(rewards)
            sleep(1. / 60)

        if SAVE_GIFS:
            gif_path = f'./pendulum_pipeline_mpc_by_sgd_{e}.gif'
            imageio.mimsave(gif_path, frames, fps=29)


if __name__ == '__main__':
    # visual_test_mpc_by_sgd()
    # performance_test(output_path='./pipeline_pendulum_exp1_100reps.pkl')
    # visual_test()
    # vae_real_vs_decoded()
    demo()
