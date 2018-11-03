from pathlib import Path
from time import sleep

import torch
import gym
from gym.envs.classic_control import PendulumEnv
import numpy as np

import _gpp
from gpp.models import MDN_Model
from gpp.mpc import MPC
from gpp.dataset import EnvDataset

BATCH_SIZE = 16
TRAINING_EPOCHS = 50
N_EPISODES = 200
EPISODE_LENGTH = 40
OVERWRITE_EXISTING = False

MDN_COMPONENTS = 5
MPC_HORIZON = 20
MPC_SEQUENCES = 2000


if __name__ == '__main__':

    if torch.cuda.is_available():
        print("CUDA available, proceeding with GPU...")
        device = torch.device("cuda")
    else:
        print("No GPU found, proceeding with CPU...")
        device = torch.device("cpu")

    env = gym.make('Pendulum-v0')
    raw_env = env.unwrapped # type: PendulumEnv
    raw_env.seed(42)
    np_random = raw_env.np_random

    n_inputs = raw_env.observation_space.low.size + raw_env.action_space.low.size
    n_outputs = raw_env.observation_space.low.size

    model = MDN_Model(n_inputs, n_outputs, MDN_COMPONENTS, np_random=np_random, device=device)

    model_path = Path('../out/tmp_mdn_model.pkl')
    data_path = Path('../out/tmp_mdn_data.pkl')

    do_train = True
    if model_path.exists():
        print('Found existing model.')
        if OVERWRITE_EXISTING:
            print('Overwriting...')
        else:
            model.load(model_path)
            do_train = False
    else:
        print('Existing model not found.')

    if do_train:

        do_generate = False
        dataset = EnvDataset(env)
        if data_path.exists():
            print('Loading data...')
            dataset.load(data_path)
            if dataset.episodes != N_EPISODES or dataset.episode_length != EPISODE_LENGTH:
                print('Existing data is not compatible with the desired parameters.')
                do_generate = True
        else:
            do_generate = True

        if do_generate:
            print('Generating data...')
            dataset.generate(N_EPISODES, EPISODE_LENGTH)
            dataset.save(data_path)
        episodes = dataset.data

        print('Training...')
        losses = model.train(episodes, TRAINING_EPOCHS, batch_size=BATCH_SIZE)

        print('Saving model...')
        model.save(model_path)

    print('Testing model...')
    controller = MPC(env, model, MPC_HORIZON, MPC_SEQUENCES, np_random)
    obs = env.reset()

    for _ in range(20000):
        env.render()
        action = controller.get_action(obs)
        obs, rewards, dones, info = env.step(action)
        sleep(1. / 60)
