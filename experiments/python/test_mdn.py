from pathlib import Path
from time import sleep

import gym
from gym.envs.classic_control import PendulumEnv
import numpy as np

import _gpp
from gpp.models import MDN_Model
from gpp.mpc import MPC

TRAINING_EPOCHS = 200
N_EPISODES = 500
EPISODE_LENGTH = 200
OVERWRITE_EXISTING = False


if __name__ == '__main__':

    env = gym.make('Pendulum-v0')
    raw_env = env.unwrapped # type: PendulumEnv
    raw_env.seed(42)
    np_random = raw_env.np_random

    n_inputs = raw_env.observation_space.low.size + raw_env.action_space.low.size
    n_components = raw_env.observation_space.low.size

    model = MDN_Model(n_inputs, n_components, np_random=np_random)
    model_path = Path('../out/tmp_mdn_model.pkl')

    do_train = True
    if model_path.exists():
        model.load(model_path)
        print('Found existing model.')
        if OVERWRITE_EXISTING:
            print('Overwriting...')
        else:
            do_train = False
    else:
        print('Existing model not found.')

    if do_train:

        print('Generating data...')
        episodes = []
        for e in range(N_EPISODES):
            obs = env.reset()
            actions = np.zeros((EPISODE_LENGTH,) + raw_env.action_space.shape)
            states = np.zeros((EPISODE_LENGTH + 1,) + raw_env.observation_space.shape)
            states[0] = obs
            for s in range(EPISODE_LENGTH):
                rand_action = raw_env.action_space.sample()
                obs, rewards, dones, info = env.step(rand_action)
                states[s+1] = obs
                actions[s] = rand_action
            episodes.append((states, actions))

        print('Training...')
        losses = model.train(episodes, TRAINING_EPOCHS)

        print('Saving model...')
        model.save(model_path)

    print('Testing model...')

    horizon = 20
    n_sequences = 2000
    controller = MPC(env, model, horizon, n_sequences, np_random)

    obs = env.reset()
    for _ in range(20000):
        env.render()
        action = controller.get_action()
        obs, rewards, dones, info = env.step(action)
        sleep(1. / 60)
