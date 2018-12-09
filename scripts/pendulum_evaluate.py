import os
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from stable_baselines import TRPO
import gym
from gym.envs.classic_control import GaussianPendulumEnv

# noinspection PyUnresolvedReferences
import _gpp
from gpp.models import MDN_Model, PendulumSim, MlpModel
from gpp.models.utilities import get_observations
from gpp.mpc import MPC

RESULTS_DIR = "../out/test_results/"
TEST_MASS_MEAN = (0.050, 1.300)
TEST_MASS_STDEV = 0.0


def _generate_test_dataset(seed=42, episodes=300) -> pd.DataFrame:

    env = gym.make('GaussianPendulum-v0')
    raw_env = env.unwrapped
    assert isinstance(raw_env, GaussianPendulumEnv)

    raw_env.configure(
        seed=seed,
        mass_mean=TEST_MASS_MEAN,
        mass_stdev=TEST_MASS_STDEV,
        embed_knowledge=True,
        perfect_knowledge=False,
        gym_env=env
    )

    np_random = np.random.RandomState(seed=seed)
    initial_states = np_random.uniform([0, -raw_env.max_speed], [2 * np.pi, raw_env.max_speed], size=(episodes, 2))

    data = []
    for e in range(episodes):
        env.reset()
        data.append(dict(
            mass_distr_params=raw_env.mass_distr_params,
            sampled_mass=raw_env.physical_props[1],
            initial_state=initial_states[e]
        ))

    return pd.DataFrame(data)


def _run_model(env: gym.Env, controller_fun, test_data: pd.DataFrame, episode_length=200,
               embed_knowledge=False, perfect_knowledge=False):

    raw_env = env.unwrapped
    assert isinstance(raw_env, GaussianPendulumEnv)

    initial_states = test_data.initial_state
    mass_distr_params = test_data.mass_distr_params
    sampled_masses = test_data.sampled_mass
    episodes = len(test_data)
    pd_index = list(test_data.index)

    # configure the environment
    raw_env.configure(
        mass_mean=np.mean(sampled_masses),
        mass_stdev=np.std(sampled_masses),
        embed_knowledge=embed_knowledge,
        perfect_knowledge=perfect_knowledge,
        gym_env=env
    )

    # Note that mass mean and stdev passed in configure() are not actually used by the environment itself,
    # as we will set our predefined values of masses. It's still useful to have some numbers in there
    # because they will be used to define the bounds of the observation space.

    # run controller_fun on environment
    rewards = np.zeros(episodes)
    for i in tqdm(range(episodes)):
        pdi = pd_index[i]

        # reset the environment to ensure that everything is clean
        env.reset()

        # impose initial state
        raw_env.state = initial_states[pdi]

        # impose sampled mass
        raw_env.sampled_mass = sampled_masses[pdi]

        # inform env about the original distribution used
        raw_env.mass_distr_params = mass_distr_params[pdi]

        observations = get_observations(raw_env)
        for t in range(episode_length):
            action = controller_fun(observations)
            observations, reward, _, _ = env.step(action)
            rewards[i] += reward

    return rewards


def _evaluation_worker(test_data, model_type, model_path, perfect_knowledge,
                       episode_length=200, model_kwargs=None):

    # init environment
    env = gym.make('GaussianPendulum-v0')
    model_kwargs = model_kwargs or dict()

    if model_type.startswith('mpc'):

        # load model
        if model_type == 'mpc-mdn':
            model = MDN_Model.load(Path(model_path), **model_kwargs)
        elif model_type == 'mpc-mlp':
            model = MlpModel.load(Path(model_path), **model_kwargs)
        elif model_type == 'mpc-sim':
            model = PendulumSim(env, **model_kwargs)
        else:
            raise NotImplementedError

        mpc = MPC(env, model, horizon=20, n_action_sequences=2000, np_random=None)

        def next_action(obs):
            return mpc.get_action(obs)

        model_info = dict(
            type=model_type,
            horizon=mpc.horizon,
            sequences=mpc.n_action_sequences,
            perfect_knowledge=perfect_knowledge
        )

    elif model_type == 'trpo':

        # load model
        model = TRPO.load(model_path, env=env, **model_kwargs)

        def next_action(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action

        model_info = dict(
            type='trpo',
            perfect_knowledge=perfect_knowledge
        )

    else:
        raise NotImplementedError

    rewards = _run_model(
        env,
        next_action,
        test_data,
        episode_length=episode_length,
        embed_knowledge=perfect_knowledge,
        perfect_knowledge=perfect_knowledge
    )

    results = pd.DataFrame(test_data)
    results = results.assign(rewards=pd.Series(rewards).values)
    results = results.assign(model_info=[model_info] * len(results))

    return results


def get_test_dataset_mean(n_episodes=50, seed=42):
    test_data = _generate_test_dataset(episodes=n_episodes, seed=seed)
    return test_data['sampled_mass'].mean()


def run_evaluation(model_type=None, model_path=None, output_suffix="", perfect_knowledge=False, workers=1, seed=42,
                   store_csv=True, n_episodes=50, episode_length=200, model_kwargs=None):

    if model_type is None:
        raise ValueError

    print('Generating test data...')
    test_data = _generate_test_dataset(episodes=n_episodes, seed=seed)
    total_runs = len(test_data)

    print(f'Running evaluation of {model_type}...')
    tick_t = timer()

    if workers > 1:
        workers = min(workers, total_runs)
        parallel = Parallel(n_jobs=workers, backend='threading', verbose=5)
        split_test_data = list([x.copy() for x in np.array_split(test_data, workers)])

        results = parallel(delayed(_evaluation_worker)(
            x, model_type, model_path, perfect_knowledge, episode_length, model_kwargs
        ) for x in split_test_data)

        results = pd.concat(results)
    else:
        results = _evaluation_worker(
            test_data, model_type, model_path, perfect_knowledge, episode_length, model_kwargs
        )

    tock_t = timer()
    print(f'Done. Took ~{round(tock_t - tick_t)}s')
    print(f'Mean reward = {results.mean()}')

    if store_csv:
        csv_path = f'{RESULTS_DIR}/{model_type}{output_suffix}.csv'
        results.to_csv(csv_path)

    return results


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_evaluation(model_type='trpo', workers=7)
    run_evaluation(model_type='mpc-mdn', output_suffix="-blind", perfect_knowledge=False, workers=7)
    run_evaluation(model_type='mpc-mdn', output_suffix="-pk", perfect_knowledge=True, workers=7)
    run_evaluation(model_type='mpc-sim', workers=7)
