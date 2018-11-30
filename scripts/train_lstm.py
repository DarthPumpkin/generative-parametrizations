from pathlib import Path
from time import sleep
from datetime import datetime

import imageio
import torch
import gym
from gym.envs.robotics import FetchPickAndPlaceSphereEnv, FetchReachEnv
import numpy as np

import _gpp
from gpp.models.lstm_model import LSTM_Model
from gpp.models.utilities import get_observation_space, get_observations
from gpp.mpc import MPC
from gpp.dataset import EnvDataset


BATCH_SIZE = 32
TRAINING_EPOCHS = 100
N_EPISODES = 4000
EPISODE_LENGTH = 100
OVERWRITE_EXISTING = True
VISUAL_TEST = True

# MDN_COMPONENTS = 6
MPC_HORIZON = 5
MPC_SEQUENCES = 50000

#ENV_ID = 'FetchPushSphereDense-v1'
#EXP_NAME = 'push_sphere_lstm_strategy'

ENV_ID = 'FetchReachDense-v1'
EXP_NAME = 'reach_lstm'


def push_strategy(raw_env, obs):
    if np.random.uniform() < 1 / 4:
        gripper_pos = obs[:3]
        object_pos = obs[3:6]
        delta = object_pos - gripper_pos
        action = np.r_[delta, 0.0] * 5.0
        return action.clip(raw_env.action_space.low, raw_env.action_space.high)
    else:
        return raw_env.action_space.sample()


STRATEGY = None


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
    n_inputs = observation_space.low.size + env.action_space.low.size
    n_outputs = observation_space.low.size

    model = LSTM_Model(n_inputs, 50, n_outputs, np_random=np_random, device=device)

    model_path = Path(f'./out/{EXP_NAME}_model.pkl')
    model_path = Path(f'./out/{EXP_NAME}_model_e20.pkl')
    data_path = Path(f'./out/{EXP_NAME}_data.pkl')

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
            dataset.generate(N_EPISODES, EPISODE_LENGTH, strategy=STRATEGY)
            dataset.save(data_path)
        episodes = dataset.data

        def epoch_callback(epoch, loss):
            print(epoch, loss)
            if epoch % 1 == 0:
                path = Path(f'./out/{EXP_NAME}_model_e{epoch}.pkl')
                model.save(path)

        print('Training...')
        losses = model.train(episodes, TRAINING_EPOCHS, batch_size=BATCH_SIZE, epoch_callback=epoch_callback,
                             scale_data=True)

        print('Saving model...')
        model.save(model_path)

    if VISUAL_TEST:
        print('Testing model...')
        controller = MPC(env, model, MPC_HORIZON, MPC_SEQUENCES, np_random)

        for e in range(2000):
            env.reset()

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
