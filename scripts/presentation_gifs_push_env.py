import time
from pathlib import Path

import cv2
import imageio
import gym
import numpy as np
import mujoco_py as mj
from gym.envs.robotics import FetchPickAndPlaceSphereEnv

from gpp.models.vision import VisionModel

SAVE_GIFS = True

PUSH_VAE_MODEL_PATH = Path('./tf_vae/kl2rl1-z16-b250-push_sphere_v0vae-fetch199.json')
PUSH_VAE_ARGS = dict(z_size=16, batch_size=32)


def init():
    env = gym.make("FetchPushSphere-v1")
    raw_env = env.unwrapped  # type: FetchPickAndPlaceSphereEnv
    sim = raw_env.sim  # type: mj.MjSim

    raw_env.target_in_the_air = False
    raw_env.mocap_bodies_visible = False

    object_name = 'object0'
    target_name = 'target0'

    object_body_i = sim.model.body_names.index(object_name)
    object_site_i = sim.model.site_names.index(object_name)
    target_site_i = sim.model.site_names.index(target_name)

    sim.model.site_size[object_site_i] *= 1.5 * 2.0
    sim.model.site_size[target_site_i] *= 1.5

    sim.model.site_rgba[object_site_i] = (1.0, 1.0, 0.0, 1.0)

    sim.model.body_mass[object_body_i] = 1

    object_geom_i = np.where(sim.model.geom_bodyid == object_body_i)[0][0]

    table_name = 'table0'
    table_body_index = sim.model.body_names.index(table_name)
    table_geom_index = np.where(sim.model.geom_bodyid == table_body_index)[0][0]

    sim.model.geom_friction[object_geom_i, :] = 0.0001
    sim.model.geom_friction[table_geom_index, :] = 0.0001

    return env


def get_action(obs, table_dist):
    gripper_pos = obs['observation'][:3]
    object_pos = obs['observation'][3:6]

    object_pos[2] += table_dist
    delta = object_pos - gripper_pos
    dir_ = delta / np.linalg.norm(delta)
    action = (np.r_[delta + dir_ * 0.5, 0.0]) * 5.0

    action += np.random.normal()
    return action


def vae_real_vs_decoded():

    env = init()
    interrupt = False

    model = VisionModel(PUSH_VAE_MODEL_PATH, z_size=16, batch_size=32)

    for e in range(10):
        obs = env.reset()
        table_dist = np.random.uniform(0.03, 0.05)

        frames = []
        for s in range(30):
            rgb_obs = env.render(mode='rgb_array')
            action = get_action(obs, table_dist) * 0.2

            vae_sight = model.vae_decode(model.vae_encode(rgb_obs), apply_filter=False)[0]
            vae_sight = cv2.resize(vae_sight, dsize=(500, 500), interpolation=cv2.INTER_LINEAR)

            comp_img = np.zeros((500, 500*2, 3), dtype=np.float32)
            comp_img[:, :500] = rgb_obs / 255.
            comp_img[:, 500:] = vae_sight

            frames.append(comp_img)

            vae_sight = cv2.cvtColor(comp_img, cv2.COLOR_RGB2BGR)
            cv2.imshow('frame', vae_sight)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                interrupt = True
                break

            obs = env.step(action)[0]
            if interrupt:
                break

        if SAVE_GIFS:
            gif_path = f'./presentation_push_vae_{e}.gif'
            imageio.mimsave(gif_path, frames, fps=29)

        if interrupt:
            break

    cv2.destroyAllWindows()


def push_simple_gif():

    env = init()
    frames = []
    for e in range(5):
        observation = env.reset()
        table_dist = np.random.uniform(0.03, 0.05)

        for i in range(16):
            action = get_action(observation, table_dist)
            frames.append(env.render(mode='rgb_array', rgb_options=dict(camera_id=3)))
            observation, reward, done, info = env.step(action)
            time.sleep(1/60.)

    if SAVE_GIFS:
        gif_path = f'./presentation_push_env.gif'
        imageio.mimsave(gif_path, frames, fps=10)


if __name__ == '__main__':
    vae_real_vs_decoded()
