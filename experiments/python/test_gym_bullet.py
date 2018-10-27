import gym
import pybullet as p
import time
import pybullet_data
from pybulletgym.envs.roboschool.gym_locomotion_envs import AntBulletEnv


if __name__ == '__main__':

    env = gym.make("AntPyBulletEnv-v0")
    env.render(mode="human")

    observation = env.reset()

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf", basePosition=[0, 0, 0.01])

    while True:
        action = env.action_space.sample()
        env.render()
        observation, reward, done, info = env.step(action)
        time.sleep(1/60.)
