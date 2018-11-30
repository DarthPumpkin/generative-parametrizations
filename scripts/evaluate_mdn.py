from pathlib import Path

import torch
import gym
import numpy as np

import _gpp
from gpp.models import MDN_Model
from gpp.dataset import EnvDataset


N_EPISODES = 100
EP_LENGTH = 20


def push_strategy(raw_env, obs):
    if np.random.uniform() < 1 / 4:
        gripper_pos = obs[:3]
        object_pos = obs[3:6]
        delta = object_pos - gripper_pos
        action = np.r_[delta, 0.0] * 5.0
        return action.clip(raw_env.action_space.low, raw_env.action_space.high)
    else:
        return raw_env.action_space.sample()


def evaluate(model_path: Path, env_id: str, strategy=None):

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
    dataset.generate(N_EPISODES, EP_LENGTH, strategy=strategy)
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


if __name__ == '__main__':
    #evaluate(Path('./out/fetch_mdn_clamp_6comp_model.pkl'), 'FetchReachDense-v1')
    evaluate(Path('./out/fetch_mdn_scaled_6comp_model.pkl'), 'FetchReachDense-v1')
    evaluate(Path('./out/push_sphere_mdn_strategy_model_e0.pkl'), 'FetchPushSphereDense-v1', strategy=push_strategy)
