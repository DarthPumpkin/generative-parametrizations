from pathlib import Path
from time import sleep
from datetime import datetime

import imageio
import torch
import gym
# from gym.envs.robotics import FetchPickAndPlaceSphereEnv, FetchReachEnv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import _gpp
from gpp.models.mlp_model import MlpModel, SimpleMlpModel
from gpp.models.utilities import get_observation_space, get_observations
from gpp.mpc import MPC
from gpp.dataset_old import EnvDataset
from gpp.reward_functions import RewardFunction
from evaluate_mdn import evaluate


BATCH_SIZE = 32
TRAINING_EPOCHS1 = 150
TRAINING_EPOCHS2 = 150
OVERWRITE_EXISTING = True
VISUAL_TEST = True

MPC_HORIZON = 20
MPC_SEQUENCES = 2000

ENV_ID = 'Pendulum-v0'
EXP_NAME = 'pendulum_v0_vision_mlp'

DATA_PATH = Path(f'../data/pendulum_v0_details.pkl')
Z_PATH = Path(f'../data/pendulum_v0_latent_kl2rl1-z6-b100.npz')
Z_TO_OBS = True
Z_SIZE = 6


def main():

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

    mlp1_n_inputs = Z_SIZE + env.action_space.low.size
    mlp1_n_outputs = Z_SIZE

    # mlp1_n_inputs = observation_space.low.size + env.action_space.low.size
    # mlp1_n_outputs = observation_space.low.size

    mlp2_n_inputs = Z_SIZE
    mlp2_n_outputs = observation_space.low.size

    mlp1 = MlpModel(mlp1_n_inputs, mlp1_n_outputs, hidden_units=(128, 128, 128, 64), np_random=np_random, device=device)
    mlp2 = SimpleMlpModel(mlp2_n_inputs, mlp2_n_outputs, hidden_units=(128, 128, 32), device=device)

    mlp1_model_path = Path(f'./out/{EXP_NAME}1_model.pkl')
    mlp2_model_path = Path(f'./out/{EXP_NAME}2_model.pkl')
    # model_path = Path(f'./out/{EXP_NAME}_model_e30.pkl')

    do_train = True
    if mlp1_model_path.exists():
        print('Found existing model.')
        if OVERWRITE_EXISTING:
            print('Overwriting...')
        else:
            print(mlp1_model_path.as_posix())
            print(mlp2_model_path.as_posix())
            exit(0)
    else:
        print('Existing model not found.')

    if do_train:

        ##########################################################

        df = pd.read_pickle(DATA_PATH)
        n_episodes = df['episode'].max() + 1
        episode_length = df['step'].max() + 1
        all_z = np.load(Z_PATH)['arr_0']

        episodes = []

        for i in range(n_episodes):
            ep_df = df[df['episode'] == i]
            actions = np.array(ep_df['raw_action'].tolist())
            raw_obs = np.array(ep_df['raw_obs'].tolist())

            # s = raw_obs
            s = all_z[i]
            episodes.append((s, actions[1:]))

        mlp2_x = all_z.reshape(n_episodes * episode_length, -1)
        mlp2_y = np.array(list(df['raw_obs']))

        ##########################################################

        def epoch_callback(epoch, loss):
            print(epoch, loss)

        print('Training...')
        losses = mlp1.train(episodes, epochs=TRAINING_EPOCHS1, batch_size=BATCH_SIZE,
                            epoch_callback=epoch_callback, scale_data=True, shuffle_data=True)
        losses = mlp2.train(mlp2_x, mlp2_y, epochs=TRAINING_EPOCHS2, batch_size=BATCH_SIZE,
                            epoch_callback=epoch_callback, scale_data=True, shuffle_data=True)

        print('Saving models...')
        mlp1.save(mlp1_model_path)
        mlp2.save(mlp2_model_path)


if __name__ == '__main__':
    main()
