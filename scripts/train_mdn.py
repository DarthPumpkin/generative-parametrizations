from pathlib import Path
from time import sleep
from datetime import datetime

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
TRAINING_EPOCHS = 100
N_EPISODES = 1800
EPISODE_LENGTH = 100
OVERWRITE_EXISTING = False
VISUAL_TEST = True

MDN_COMPONENTS = 6
MPC_HORIZON = 5
MPC_SEQUENCES = 26580

#ENV_ID = 'FetchPushSphereDense-v1'
#EXP_NAME = 'push_sphere_mdn_strategy'

ENV_ID = 'FetchReachDense-v1'
EXP_NAME = 'reach_mdn'


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

    model = MDN_Model(n_inputs, n_outputs, MDN_COMPONENTS,
                      hidden_units=(20,), np_random=np_random, device=device)

    model_path = Path(f'./out/{EXP_NAME}_model.pkl')
    # model_path = Path(f'./out/{EXP_NAME}_model_e10.pkl')
    data_path = Path(f'./out/{EXP_NAME}_data.pkl')

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
            dataset.generate(N_EPISODES, EPISODE_LENGTH, strategy=STRATEGY)
            dataset.save(data_path)
        episodes = dataset.data

        def epoch_callback(epoch, loss):
            print(epoch, loss)
            if epoch % 10 == 0:
                path = Path(f'./out/{EXP_NAME}_model_e{epoch}.pkl')
                model.save(path)
                evaluate_mdn(path, ENV_ID, strategy=STRATEGY)

        if False:
            eps_with_changes = 0
            for e in episodes:
                prev_pos = np.zeros(3)
                changed = -1
                for s in e[0]:
                    sphere_pos = s[3:6].copy()
                    changed += int(not np.allclose(sphere_pos, prev_pos, atol=0.001))
                    prev_pos = sphere_pos
                eps_with_changes += int(changed > 0)
            print(eps_with_changes)

        print('Training...')
        losses = model.train(episodes, TRAINING_EPOCHS, batch_size=BATCH_SIZE, epoch_callback=epoch_callback,
                             scale_data=True, shuffle_data=True)

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
