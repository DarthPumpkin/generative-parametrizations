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


BATCH_SIZE = 16
TRAINING_EPOCHS = 100
N_EPISODES = 4000
EPISODE_LENGTH = 10
OVERWRITE_EXISTING = False
VISUAL_TEST = True

# MDN_COMPONENTS = 6
MPC_HORIZON = 5
MPC_SEQUENCES = 13

ENV_ID = 'FetchPushSphereDense-v1'
EXP_NAME = 'push_sphere_v0_lstm'

# ENV_ID = 'FetchReachDense-v1'
# EXP_NAME = 'reach_lstm'


def push_strategy(raw_env, obs):
    if np.random.uniform() < 1 / 4:
        gripper_pos = obs[:3]
        object_pos = obs[3:6]
        delta = object_pos - gripper_pos
        dir_ = delta / np.linalg.norm(delta)
        action = np.r_[delta + dir_*0.5, 0.0] * 5.0
        return action.clip(raw_env.action_space.low, raw_env.action_space.high)
    else:
        return raw_env.action_space.sample()


STRATEGY = push_strategy
STRATEGY_PERIOD = 10


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

    model = LSTM_Model(n_inputs, 64, n_outputs, n_layers=2, np_random=np_random, device=device, window_size=5)

    model_path = Path(f'./out/{EXP_NAME}_model.pkl')
    # model_path = Path(f'./out/{EXP_NAME}_model_e30.pkl')
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



        df = pd.read_pickle('../data/push_sphere_v0_details.pkl')
        episode_length = df['step'].max() + 1
        n_episodes = df['episode'].max() + 1

        episodes = []
        for i in range(n_episodes):
            ep_df = df[df['episode'] == i]
            actions = np.array(ep_df['raw_action'].tolist())
            states = np.array(ep_df['raw_obs'].tolist())
            episodes.append((states, actions[1:]))



        model.use_window_in_forward_sim = True


        if False:
            dataset = EnvDataset(env)
            dataset.generate(n_episodes, episode_length)
            episodes = dataset.data



        def epoch_callback(epoch, loss):
            print(epoch, loss)
            if epoch % 10 == 0:
                path = Path(f'./out/{EXP_NAME}_model_e{epoch}.pkl')
                model.save(path)
                evaluate(LSTM_Model, path, 'FetchPushSphereDense-v1', strategy=push_strategy,
                         strategy_period=episode_length, episodes=episodes)
                # evaluate(LSTM_Model, path, 'FetchReachDense-v1')

        print('Training...')
        losses = model.train(episodes, epochs=TRAINING_EPOCHS, batch_size=BATCH_SIZE, epoch_callback=epoch_callback,
                             scale_data=True, scale_targets=True)

        print('Saving model...')
        model.save(model_path)

    if VISUAL_TEST:
        print('Testing model...')

        reward_fn = RewardFunction.simplified_push_reward(env)
        controller = MPC(env, model, MPC_HORIZON, MPC_SEQUENCES, np_random, reward_function=reward_fn, use_history=True)

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
