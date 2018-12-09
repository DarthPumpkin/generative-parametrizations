from pathlib import Path
from time import sleep

from tqdm import tqdm
import itertools as it
import imageio
import torch
import gym
from gym.envs.classic_control import GaussianPendulumEnv
import numpy as np
import pandas as pd

# noinspection PyUnresolvedReferences
import _gpp
from gpp.models import MDN_Model, MlpModel
from gpp.dataset_old import EnvDataset
from evaluate_mdn_mlp import evaluate as evaluate_mse
from pendulum_evaluate import run_evaluation as evaluate_gym_perf
from pendulum_evaluate import TEST_MASS_MEAN, TEST_MASS_STDEV


TRAINING_BATCH_SIZE = 16
TRAINING_EPOCHS = 20
TRAINING_EPISODES = 800
TRAINING_EPISODE_LENGTH = 50

MDN_COMPONENTS = 5
MLP_HIDDEN_UNITS = (20,)

TEST_EPISODES = 50
TEST_EPISODE_LENGTH = 200

USE_MLP = True

if USE_MLP:
    MODEL_PATH_PREFIX = './out/pendulum_motivating_mlp_'
    RESULTS_PATH = Path(f'./pendulum_motivating_mlp_results.pkl')
else:
    MODEL_PATH_PREFIX = './out/pendulum_motivating_'
    RESULTS_PATH = Path(f'./pendulum_motivating_results.pkl')

N_ITERS = 5

SETTINGS = dict(
    blind_fixed_1000g_mass=dict(
        mass_stdev=0.0,
        mass_mean=1.0,
        embed_knowledge=False,
        perfect_knowledge=False
    ),
    blind_fixed_300g_mass=dict(
        mass_stdev=0.0,
        mass_mean=0.300,
        embed_knowledge=False,
        perfect_knowledge=False
    ),
    blind_var_mass=dict(
        mass_stdev=TEST_MASS_STDEV,
        mass_mean=TEST_MASS_MEAN,
        embed_knowledge=False,
        perfect_knowledge=False
    ),
    informed_var_mass=dict(
        mass_stdev=TEST_MASS_STDEV,
        mass_mean=TEST_MASS_MEAN,
        embed_knowledge=True,
        perfect_knowledge=True
    )
)


def init_env(config):
    env = gym.make('GaussianPendulum-v0')
    raw_env = env.unwrapped  # type: GaussianPendulumEnv
    raw_env.configure(
        seed=42,
        gym_env=env,
        **config
    )
    return env


def train(model_path: Path, env: gym.Env, device=torch.device('cpu')):

    raw_env = env.unwrapped # type: GaussianPendulumEnv
    np_random = raw_env.np_random
    n_inputs = raw_env.observation_space.low.size + raw_env.action_space.low.size
    n_outputs = raw_env.observation_space.low.size

    if USE_MLP:
        model = MlpModel(n_inputs, n_outputs, MLP_HIDDEN_UNITS, np_random=np_random, device=device)
    else:
        model = MDN_Model(n_inputs, n_outputs, MDN_COMPONENTS, np_random=np_random, device=device)

    dataset = EnvDataset(env)
    dataset.generate(TRAINING_EPISODES, TRAINING_EPISODE_LENGTH)
    episodes = dataset.data

    losses = model.train(episodes, epochs=TRAINING_EPOCHS, batch_size=TRAINING_BATCH_SIZE,
                         shuffle_data=True, scale_data=True)

    model.save(model_path)

    return losses


def main():

    prev_df = None
    if RESULTS_PATH.exists():
        prev_df = pd.read_pickle(RESULTS_PATH)

    if torch.cuda.is_available():
        print("CUDA available, proceeding with GPU...")
        device = torch.device("cuda")
    else:
        print("No GPU found, proceeding with CPU...")
        device = torch.device("cpu")

    model_type = MlpModel if USE_MLP else MDN_Model
    model_type_str = 'mpc-mlp' if USE_MLP else 'mpc-mdn'

    results = []
    for (s_name, s), i in tqdm(it.product(SETTINGS.items(), range(N_ITERS)), desc='ITER'):

        if prev_df is not None:
            if prev_df[(prev_df['iteration'] == i) & (prev_df['setting'] == s_name)].shape[0] > 0:
                print(f'Setting {s_name}_{i} already present. Skipping...')
                continue

        s_mod = dict(s)
        path = Path(f'{MODEL_PATH_PREFIX}{s_name}_{i}')

        print('==> Training model...')
        env = init_env(s_mod)
        losses = train(model_path=path, env=env, device=device)

        s_mod['mass_mean'] = TEST_MASS_MEAN
        s_mod['mass_stdev'] = TEST_MASS_STDEV
        perfect_knowledge = s_mod['perfect_knowledge']

        print('==> Evaluating model MSE...')
        env = init_env(s_mod)
        mse = evaluate_mse(model_type, model_path=path, env=env, device=device)

        print('==> Evaluating performance on environment...')
        perf_results = evaluate_gym_perf(model_type_str, model_path=path, perfect_knowledge=perfect_knowledge,
                                         workers=7, seed=42, store_csv=False, n_episodes=TEST_EPISODES,
                                         episode_length=TEST_EPISODE_LENGTH, model_kwargs=dict(device=device))

        rewards = np.array(perf_results['rewards'])
        masses = np.array(perf_results['sampled_mass'])

        results.append(dict(
            setting=s_name,
            iteration=i,
            losses=losses,
            test_mse=mse,
            test_reward_mean=rewards.mean(),
            test_reward_stdev=rewards.std(),
            masses=masses
        ))

    results = pd.DataFrame(results)

    if prev_df is not None:
        results = pd.concat((prev_df, results))
        results = results.reset_index(drop=True)

    results.to_pickle(RESULTS_PATH)
    print(results)


if __name__ == '__main__':
    main()

    df = pd.read_pickle(RESULTS_PATH)
    print()
