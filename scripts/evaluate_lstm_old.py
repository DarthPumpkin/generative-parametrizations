from pathlib import Path

import torch
import gym
import numpy as np

import _gpp
from gpp.models import MDN_Model, LSTM_Model
from gpp.dataset_old import EnvDataset


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


def evaluate(model_type, model_path: Path, env_id: str, strategy=None, strategy_period=1, state_filter=None, episodes=None):

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

    model = model_type.load(model_path, device)

    if episodes is None:
        dataset = EnvDataset(env)
        print('Generating data...')
        np.random.seed(300)
        dataset.generate(N_EPISODES, EP_LENGTH, strategy=strategy, strategy_period=strategy_period)
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
            if state_filter is not None:
                err = err[state_filter]
            mse = np.mean(err**2.0)
            ms_errors[i, j] = mse

    avg_mse = ms_errors.mean()
    print(f'Avg. MSE: {avg_mse}')
    return avg_mse


if __name__ == '__main__':
    #evaluate(MDN_Model, Path('./out/fetch_mdn_clamp_6comp_model.pkl'), 'FetchReachDense-v1')
    #evaluate(MDN_Model, Path('./out/fetch_mdn_scaled_6comp_model.pkl'), 'FetchReachDense-v1')
    #evaluate(MDN_Model, Path('./out/reach_mdn_model_e10.pkl'), 'FetchReachDense-v1', strategy=None, state_filter=[0, 1, 2])
    #evaluate(MDN_Model, Path('./out/reach_mdn_model_e10.pkl'), 'FetchReachDense-v1', strategy=None, state_filter=[0, 1, 2])
    #evaluate(MDN_Model, Path('./out/reach_mdn_model_e10.pkl'), 'FetchReachDense-v1', strategy=None, state_filter=[0, 1, 2])
    evaluate(LSTM_Model, Path(f'./out/reach_lstm_model_e20.pkl'), 'FetchReachDense-v1')
    evaluate(LSTM_Model, Path(f'./out/push_sphere_lstm_strategy_short_model.pkl'), 'FetchPushSphereDense-v1',
             strategy=push_strategy, state_filter=[3, 4, 5])
