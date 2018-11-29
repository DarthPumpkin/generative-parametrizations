import gym
import time
import numpy as np
import mujoco_py as mj
from gym.envs.robotics import FetchPickAndPlaceSphereEnv


if __name__ == '__main__':

    env = gym.make("FetchPickAndPlaceSphere-v1")
    raw_env = env.unwrapped # type: FetchPickAndPlaceSphereEnv
    sim = raw_env.sim # type: mj.MjSim

    raw_env.target_in_the_air = False
    raw_env.mocap_bodies_visible = False

    sim.model.opt.gravity[0] = 2.0
    sim.model.opt.gravity[1] = 0.2

    object_name = 'object0'
    target_name = 'target0'

    object_site_i = sim.model.site_names.index(object_name)
    target_site_i = sim.model.site_names.index(target_name)

    sim.model.site_size[object_site_i] *= 1.5 * 2.0
    sim.model.site_size[target_site_i] *= 1.5

    sim.model.site_rgba[object_site_i] = (1.0, 1.0, 0.0, 1.0)

    # this doesn't seem to work sadly!
    # sim.model.geom_size[23] *= 2.5

    observation = env.reset()
    while True:
        action = env.action_space.sample()
        for i in range(60*2):
            env.render()
            observation, reward, done, info = env.step(action)
            time.sleep(1/60.)
