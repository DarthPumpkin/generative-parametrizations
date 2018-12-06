from pathlib import Path
from time import sleep
from datetime import datetime

import imageio
import torch
import gym
# from gym.envs.robotics import FetchPickAndPlaceSphereEnv, FetchReachEnv
import numpy as np
import pandas as pd

import _gpp
from gpp.models.lstm_model import LSTM_Model
from gpp.models.utilities import get_observation_space, get_observations
from gpp.mpc import MPC
from gpp.dataset_old import EnvDataset
from gpp.reward_functions import RewardFunction
from evaluate_mdn import evaluate


BATCH_SIZE = 32
TRAINING_EPOCHS = 40
OVERWRITE_EXISTING = True
VISUAL_TEST = True

MPC_HORIZON = 20
MPC_SEQUENCES = 40000

ENV_ID = 'Pendulum-v0'
EXP_NAME = 'pendulum_v0_vision_lstm'

DATA_PATH = Path(f'../data/pendulum_v0_details.pkl')
Z_PATH = Path(f'../data/pendulum_v0_latent_kl2rl1-z16-b100.npz')
Z_TO_OBS = True


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
    n_inputs = 16 + env.action_space.low.size
    n_outputs = observation_space.low.size

    model = LSTM_Model(n_inputs, 32, n_outputs, n_layers=2, np_random=np_random, device=device, window_size=5)

    model_path = Path(f'./out/{EXP_NAME}_model.pkl')
    # model_path = Path(f'./out/{EXP_NAME}_model_e30.pkl')

    do_train = True
    if model_path.exists():
        print('Found existing model.')
        if OVERWRITE_EXISTING:
            print('Overwriting...')
        else:
            model = LSTM_Model.load(model_path, device)
            do_train = False
    else:
        print('Existing model not found.')

    if do_train:

        ##########################################################

        df = pd.read_pickle(DATA_PATH)
        n_episodes = df['episode'].max() + 1
        # episode_length = df['step'].max() + 1

        episodes = []
        targets, all_z = None, None

        if Z_TO_OBS:
            all_z = np.load(Z_PATH)['arr_0']
            targets = []

        for i in range(n_episodes):
            ep_df = df[df['episode'] == i]
            actions = np.array(ep_df['raw_action'].tolist())
            raw_obs = np.array(ep_df['raw_obs'].tolist())

            if Z_TO_OBS:
                targets.append(raw_obs[1:])
                z = all_z[i]
                episodes.append((z, actions[1:]))
            else:
                episodes.append((raw_obs, actions[1:]))

        if targets is not None:
            targets = np.array(targets)

        ##########################################################

        def epoch_callback(epoch, loss):
            print(epoch, loss)
            if epoch % 10 == 0:
                path = Path(f'./out/{EXP_NAME}_model_e{epoch}.pkl')
                model.save(path)
                # evaluate(LSTM_Model, path, ENV_ID)

        print('Training...')
        losses = model.train(episodes, targets=targets, epochs=TRAINING_EPOCHS, batch_size=BATCH_SIZE,
                             epoch_callback=epoch_callback, scale_data=True, scale_targets=True)

        print('Saving model...')
        model.save(model_path)

    if VISUAL_TEST:
        print('Testing model...')

        controller = MPC(env, model, MPC_HORIZON, MPC_SEQUENCES, np_random)

        for e in range(2000):
            env.reset()
            controller.forget_history()

            for s in range(100):
                env.render()

                tic = datetime.now()

                obs = get_observations(env)
                action = controller.get_action(obs)
                _, rewards, dones, info = env.step(action)

                toc = datetime.now()
                print((toc - tic).total_seconds())
                #print(rewards)
                #sleep(1. / 60)


if __name__ == '__main__':
    main()
