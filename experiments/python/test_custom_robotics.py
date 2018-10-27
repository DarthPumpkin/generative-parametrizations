import gym
import time
import numpy as np
import mujoco_py as mj


if __name__ == '__main__':

    env = gym.make("FetchPickAndPlaceSphere-v1")
    sim = env.env.sim # type: mj.MjSim

    sim.model.opt.gravity[0] = 2.0
    sim.model.opt.gravity[1] = 0.2

    observation = env.reset()
    while True:
        action = env.action_space.sample()
        for i in range(60*2):
            env.render()
            observation, reward, done, info = env.step(action)
            time.sleep(1/60.)
