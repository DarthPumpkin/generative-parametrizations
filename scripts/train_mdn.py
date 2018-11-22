from pathlib import Path
from time import sleep

import imageio
import torch
import gym
from gym.envs.robotics import FetchPickAndPlaceSphereEnv, FetchReachEnv
import numpy as np

import _gpp
from gpp.models import MDN_Model
from gpp.models.utilities import get_observation_space, get_observations
from gpp.mpc import MPC
from gpp.dataset import EnvDataset
from evaluate_mdn import evaluate as evaluate_mdn


BATCH_SIZE = 32
TRAINING_EPOCHS = 6
N_EPISODES = 1800
EPISODE_LENGTH = 100
OVERWRITE_EXISTING = False
SAVE_GIFS = False

MDN_COMPONENTS = 6
MPC_HORIZON = 5
MPC_SEQUENCES = 18000

ENV_ID = 'FetchReachDense-v1'
EXP_NAME = 'fetch_mdn_scaled_6comp'


if __name__ == '__main__':

    if torch.cuda.is_available():
        print("CUDA available, proceeding with GPU...")
        device = torch.device("cuda")
    else:
        print("No GPU found, proceeding with CPU...")
        device = torch.device("cpu")

    env = gym.make(ENV_ID)
    raw_env = env.unwrapped

    np_random = raw_env.np_random
    observation_space = get_observation_space(env)
    n_inputs = observation_space.low.size + env.action_space.low.size
    n_outputs = observation_space.low.size

    model = MDN_Model(n_inputs, n_outputs, MDN_COMPONENTS, np_random=np_random, device=device)

    model_path = Path(f'./out/{EXP_NAME}_model.pkl')
    #data_path = Path(f'./out/{EXP_NAME}_data.pkl')
    data_path = Path(f'./out/fetch_mdn_data2.pkl')

    do_train = True
    if model_path.exists():
        print('Found existing model.')
        if OVERWRITE_EXISTING:
            print('Overwriting...')
        else:
            model = MDN_Model.load(model_path, device)
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

        def epoch_callback(epoch, loss):
            print(epoch, loss)
            path = Path(f'./out/{EXP_NAME}_model_e{epoch}.pkl')
            model.save(path)
            evaluate_mdn(path, ENV_ID)

        print('Training...')
        losses = model.train(episodes, TRAINING_EPOCHS, batch_size=BATCH_SIZE, epoch_callback=epoch_callback, scale_data=True)

        print('Saving model...')
        model.save(model_path)

    print('Testing model...')
    controller = MPC(env, model, MPC_HORIZON, MPC_SEQUENCES, np_random)

    for e in range(2000):
        env.reset()

        for s in range(100):
            env.render()
            obs = get_observations(env)
            action = controller.get_action(obs)
            _, rewards, dones, info = env.step(action)
            print(rewards)
            #sleep(1. / 60)
