from pathlib import Path
from time import sleep

import numpy as np

import gym
from gym.envs.classic_control import GaussianPendulumEnv

import _gpp
from gpp.mpc import MPC
from gpp.models.vision import VaeLstmTFModel
from reward_functions import _build_simplified_reward_fn_push_env
from generate_rgb_obs import _reset_push_sphere_v0

l2l_model_path = Path('./out/push_sphere_v1_l2lmodel.hdf5')
# l2reward_model_path = Path('./out/l2rewardmode-reward_exp2_coeff1.h5')
l2reward_model_path = Path('./out/l2rewardmodel.h5')
VAE_MODEL_PATH = Path('./tf_vae/kl2rl1-z16-b250-push_sphere_v0vae-fetch199.json')

VAE_ARGS = dict(z_size=16, batch_size=32)

MPC_HORIZON = 4
MPC_SEQUENCES = 5_000


def _setup_fetch_sphere_big(env):
    raw_env = env.unwrapped
    model = raw_env.sim.model
    raw_env.target_in_the_air = False
    raw_env.mocap_bodies_visible = False
    object_site_i = model.site_names.index('object0')
    target_site_i = model.site_names.index('target0')
    model.site_size[object_site_i] *= 1.5 * 2.0
    model.site_size[target_site_i] *= 1.5


def _setup_fetch_sphere_big_longer(env):
    raw_env = env.unwrapped
    raw_env.block_gripper = True
    _setup_fetch_sphere_big(env)


def main():

    env = gym.make('FetchPushSphereDense-v1')

    raw_env = env.unwrapped # type: GaussianPendulumEnv
    raw_env.seed(42)
    np_random = raw_env.np_random

    _setup_fetch_sphere_big_longer(env)

    reward_fn = _build_simplified_reward_fn_push_env(coeff=0.0, exp=1.0)

    model = VaeLstmTFModel(VAE_MODEL_PATH, l2l_model_path=l2l_model_path,
                           l2reward_model_path=l2reward_model_path, **VAE_ARGS)
    controller = MPC(env, model, MPC_HORIZON, MPC_SEQUENCES,
                     np_random, use_history=True, direct_reward=True,
                     action_period=1)

    for e in range(2000):
        env_obs = env.reset()
        _reset_push_sphere_v0(env)
        controller.forget_history()
        for s in range(100):
            rgb_obs = env.render(mode='rgb_array', rgb_options=dict(camera_id=3))
            action = controller.get_action(rgb_obs)

            env_obs, rewards, dones, info = env.step(action)

            raw_obs = env_obs['observation']
            goal = env_obs['desired_goal']
            mod_rew = reward_fn(raw_obs, goal=goal).item()

            print(f'Actual reward: {mod_rew}')
            # sleep(1. / 60)


if __name__ == '__main__':
    main()
