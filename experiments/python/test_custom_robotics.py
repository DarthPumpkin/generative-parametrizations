import gym
import time
import numpy as np
import mujoco_py as mj
from gym.envs.robotics import FetchPickAndPlaceSphereEnv


if __name__ == '__main__':

    env = gym.make("FetchPushSphere-v1")
    raw_env = env.unwrapped # type: FetchPickAndPlaceSphereEnv
    sim = raw_env.sim # type: mj.MjSim

    raw_env.target_in_the_air = False
    raw_env.mocap_bodies_visible = False

    #sim.model.opt.gravity[0] = 2.0
    #sim.model.opt.gravity[1] = 0.2

    object_name = 'object0'
    target_name = 'target0'

    object_body_i = sim.model.body_names.index(object_name)
    object_site_i = sim.model.site_names.index(object_name)
    target_site_i = sim.model.site_names.index(target_name)

    sim.model.site_size[object_site_i] *= 1.5 * 2.0
    sim.model.site_size[target_site_i] *= 1.5

    sim.model.site_rgba[object_site_i] = (1.0, 1.0, 0.0, 1.0)

    sim.model.body_mass[object_body_i] = 50

    object_geom_i = np.where(sim.model.geom_bodyid == object_body_i)[0][0]

    table_name = 'table0'
    table_body_index = sim.model.body_names.index(table_name)
    table_geom_index = np.where(sim.model.geom_bodyid == table_body_index)[0][0]

    sim.model.geom_friction[object_geom_i, :] = 0.0001
    sim.model.geom_friction[table_geom_index, :] = 0.0001

    # this doesn't seem to work sadly!
    # sim.model.geom_size[23] *= 2.5


    while True:
        observation = env.reset()
        #action = env.action_space.sample()
        go_to_ball = np.random.uniform() < 3/4
        if not go_to_ball:
            action = env.action_space.sample()
        for i in range(16):
            if go_to_ball:
                gripper_pos = observation['observation'][:3]
                object_pos = observation['observation'][3:6]
                delta = object_pos - gripper_pos
                dir_ = delta / np.linalg.norm(delta)
                action = (np.r_[delta + dir_*0.5, 0.0]) * 5.0

            env.render()
            observation, reward, done, info = env.step(action)
            time.sleep(1/60.)
