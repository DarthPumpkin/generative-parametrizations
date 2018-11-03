import gym
from time import sleep

import _gpp
from gpp.mpc import MPC
from gpp.models import PendulumSim, CartPoleSim


if __name__ == '__main__':

    # env = gym.make('Pendulum-v0')
    env = gym.make('CartPole-v0')

    raw_env = env.unwrapped
    raw_env.seed(42)
    np_random = raw_env.np_random

    # model = PendulumSim(env, np_random=np_random)
    model = CartPoleSim(env, np_random=np_random)

    horizon = 100
    n_sequences = 2000
    controller = MPC(env, model, horizon, n_sequences, np_random)

    try:
        ep_length = env.spec.max_episode_steps
    except AttributeError as e:
        print('Current environment doesn\'t specify the maximum length for each episode. Using default value.')
        ep_length = 200

    for e in range(2000):
        obs = env.reset()
        for s in range(ep_length):
            env.render()
            action = controller.get_action(obs)
            obs, rewards, dones, info = env.step(action)
            sleep(1. / 60)
