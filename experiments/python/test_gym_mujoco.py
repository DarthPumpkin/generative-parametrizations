import gym
import time
import numpy as np
import mujoco_py as mj

if __name__ == '__main__':

    env = gym.make("FetchPickAndPlace-v1")
    sim = env.env.sim # type: mj.MjSim

    observation = env.reset()

    #sim.model.opt.gravity[1] = 1.0

    cube_name = 'object0'
    cube_body_index = sim.model.body_names.index(cube_name)
    cube_geom_index = np.where(sim.model.geom_bodyid == cube_body_index)[0][0]

    table_name = 'table0'
    table_body_index = sim.model.body_names.index(table_name)
    table_geom_index = np.where(sim.model.geom_bodyid == table_body_index)[0][0]

    sim.model.geom_friction[cube_geom_index, :] = 0.1
    sim.model.geom_friction[table_geom_index, :] = 0.1

    sim.model.body_mass[cube_body_index] = 0.5

    cube_pose = sim.data.get_joint_qpos(f'{cube_name}:joint').copy()
    # cube_pose[2] += 0.3
    sim.data.set_joint_qpos(f'{cube_name}:joint', cube_pose)

    while True:
        action = env.action_space.sample()

        for i in range(60*2):
            sim.model.opt.gravity[1] -= np.random.random() * 2 - 1.0
            sim.model.opt.gravity[0] -= np.random.random() * 2 - 1.0
            env.render()
            observation, reward, done, info = env.step(action)
            time.sleep(1/60.)
