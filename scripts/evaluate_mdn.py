from pathlib import Path

import torch
import gym
import numpy as np

import _gpp
from gpp.models import MDN_Model
from gpp.dataset import EnvDataset


N_EPISODES = 1000


def evaluate(model_path: Path, env_id: str):

    if torch.cuda.is_available():
        print("CUDA available, proceeding with GPU...")
        device = torch.device("cuda")
    else:
        print("No GPU found, proceeding with CPU...")
        device = torch.device("cpu")

    env = gym.make(env_id)
    env.seed(300)

    if not model_path.exists():
        print('Existing model not found.')
        raise FileNotFoundError

    model = MDN_Model.load(model_path, device)

    dataset = EnvDataset(env)
    print('Generating data...')
    dataset.generate(N_EPISODES, 1)
    episodes = dataset.data

    print('Evaluating model...')
    ms_errors = np.zeros((len(episodes), 1))
    for i, ep in enumerate(episodes):
        states, actions = ep
        actions = np.expand_dims(actions, axis=0)
        init_state = states[0]
        next_states = states[1:]
        pred_states = model.forward_sim(actions, init_state)[0]
        err = next_states - pred_states
        mse = np.mean(err**2.0)
        ms_errors[i] = mse

    avg_mse = ms_errors.mean()
    print(f'Avg. MSE: {avg_mse}')
    return avg_mse


if __name__ == '__main__':
    evaluate(Path('./out/fetch_mdn_clamp_6comp_model.pkl'), 'FetchReachDense-v1')
    evaluate(Path('./out/fetch_mdn_scaled_6comp_model.pkl'), 'FetchReachDense-v1')
