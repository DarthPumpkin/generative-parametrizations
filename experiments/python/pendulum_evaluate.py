import os
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
from joblib import Parallel, delayed
from stable_baselines import TRPO
import gym
import pandas as pd
# noinspection PyUnresolvedReferences
import _gpp
from gpp.models import MDN_Model, PendulumSim
from gpp import run_model_on_pendulum
from gpp.mpc import MPC

TEST_EPISODES = 50

PK_MDN_MODEL_PATH = "../out/pk_mdn_model.pkl"
BLIND_MDN_MODEL_PATH = "../out/blind_mdn_model.pkl"
TRPO_MODEL_PATH = "../out/tmp_trpo_model.pkl"
RESULTS_DIR = "../out/test_results/"


def _evaluation_worker(test_data, model_type, perfect_knowledge):

    # init environment
    env = gym.make('GaussianPendulum-v0')

    if model_type.startswith('mpc'):

        # load model
        if model_type == 'mpc-mdn':
            path = PK_MDN_MODEL_PATH if perfect_knowledge else BLIND_MDN_MODEL_PATH
            model = MDN_Model.load(Path(path))
        elif model_type == 'mpc-sim':
            model = PendulumSim(env)
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
        model = TRPO.load(TRPO_MODEL_PATH, env=env)

        def next_action(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action

        model_info = dict(
            type='trpo',
            perfect_knowledge=perfect_knowledge
        )

    else:
        raise NotImplementedError

    rewards = run_model_on_pendulum.test(
        env,
        next_action,
        test_data,
        episode_length=200,
        embed_knowledge=perfect_knowledge,
        perfect_knowledge=perfect_knowledge
    )

    results = pd.DataFrame(test_data)
    results = results.assign(rewards=pd.Series(rewards).values)
    results = results.assign(model_info=[model_info] * len(results))

    return results


def run_evaluation(model_type=None, output_suffix="", perfect_knowledge=False, workers=1, seed=42):

    if model_type is None:
        raise ValueError

    print('Generating test data...')
    test_data = run_model_on_pendulum.generate_test_dataset(episodes=TEST_EPISODES, seed=seed)
    total_runs = len(test_data)

    print(f'Running evaluation of {model_type}...')
    tick_t = timer()

    if workers > 1:
        workers = min(workers, total_runs)
        parallel = Parallel(n_jobs=workers, backend='multiprocessing', verbose=5)
        split_test_data = list([x.copy() for x in np.array_split(test_data, workers)])
        results = parallel(delayed(_evaluation_worker)(x, model_type, perfect_knowledge) for x in split_test_data)
        results = pd.concat(results)
    else:
        results = _evaluation_worker(test_data, model_type, perfect_knowledge)

    tock_t = timer()
    print(f'Done. Took ~{round(tock_t - tick_t)}s')
    print(f'Mean reward = {results.mean()}')

    csv_path = f'{RESULTS_DIR}/{model_type}{output_suffix}.csv'
    results.to_csv(csv_path)


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # run_evaluation(model_type='trpo', workers=7)
    # run_evaluation(model_type='mpc-mdn', output_suffix="-blind", perfect_knowledge=False, workers=7)
    # run_evaluation(model_type='mpc-mdn', output_suffix="-pk", perfect_knowledge=True, workers=7)
    # run_evaluation(model_type='mpc-sim', workers=7)
