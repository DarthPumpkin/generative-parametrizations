from pathlib import Path

import torch
import gym
import numpy as np

import _gpp
from gpp.dataset_old import EnvDataset


N_EPISODES = 100
EP_LENGTH = 20


def evaluate(model_type, model_path: Path, env: gym.Env, device=None):

    if device is None:
        if torch.cuda.is_available():
            print("CUDA available, proceeding with GPU...")
            device = torch.device("cuda")
        else:
            print("No GPU found, proceeding with CPU...")
            device = torch.device("cpu")

    env.seed(300)

    if not model_path.exists():
        print('Existing model not found.')
        raise FileNotFoundError

    model = model_type.load(model_path, device)

    dataset = EnvDataset(env)
    print('Generating data...')
    np.random.seed(300)
    dataset.generate(N_EPISODES, EP_LENGTH)
    episodes = dataset.data

    print('Evaluating model...')
    ms_errors = np.zeros((len(episodes), EP_LENGTH))
    for i, ep in enumerate(episodes):
        states, actions = ep
        for j in range(len(states)-1):
            a = actions[None, None, j]
            init_state = states[j]
            next_state = states[j+1]
            pred_state = model.forward_sim(a, init_state)[0, 0]
            err = next_state - pred_state
            mse = np.mean(err**2.0)
            ms_errors[i, j] = mse

    avg_mse = ms_errors.mean()
    print(f'Avg. MSE: {avg_mse}')
    return avg_mse
