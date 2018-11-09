from cv2 import resize, INTER_AREA

import numpy as np
import gym
from gym.envs.classic_control import PendulumEnv


def generate(episodes=200, episode_length=5):

    env = gym.make('Pendulum-v0')
    raw_env = env.unwrapped  # type: PendulumEnv

    data = []

    for e in range(episodes):
        env.reset()
        for s in range(episode_length):
            img = env.render(mode='rgb_array')
            img = resize(img, dsize=(256, 256), interpolation=INTER_AREA)
            data.append(img)
            rand_action = raw_env.action_space.sample()
            env.step(rand_action)

    data = np.array(data)
    np.save('../data/pendulum_imgs', data)

    print('Done.')


if __name__ == '__main__':
    generate()
