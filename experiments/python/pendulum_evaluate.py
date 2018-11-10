from pathlib import Path
from timeit import default_timer as timer

import numpy as np
from joblib import Parallel, delayed
from stable_baselines import TRPO
import gym
import pandas as pd
# noinspection PyUnresolvedReferences
import _gpp
from gpp.models import MDN_Model
from gpp import run_model_on_pendulum
from gpp.mpc import MPC

TEST_RUNS = 50
MODEL_TYPE = 'mdn'

if MODEL_TYPE == 'trpo':
    MODEL_PATH = "../out/tmp_trpo_model.pkl"
    SAVE_PATH = "../out/tmp_trpo_model.csv"
elif MODEL_TYPE == 'mdn':
    MODEL_PATH = "../out/tmp_mdn_model.pkl"
    SAVE_PATH = "../out/tmp_mdn_model.csv"


def _evaluation_worker(test_data):

    model_type = MODEL_TYPE

    # init environment
    env = gym.make('GaussianPendulum-v0')

    if model_type == 'mdn':

        # load model
        mdn = MDN_Model.load(Path(MODEL_PATH))
        mpc = MPC(env, mdn, horizon=20, n_action_sequences=2000, np_random=None)

        def next_action(obs):
            return mpc.get_action(obs)

        model_info = dict(
            type='mpc-mdn',
            horizon=mpc.horizon,
            sequences=mpc.n_action_sequences
        )

    elif model_type == 'trpo':

        # load model
        model = TRPO.load(MODEL_PATH, env=env)

        def next_action(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action

        model_info = dict(
            type='trpo'
        )

    else:
        raise NotImplementedError


    rewards = run_model_on_pendulum.test(env, next_action, test_data, episode_length=200)

    results = pd.DataFrame(test_data)
    results = results.assign(rewards=pd.Series(rewards).values)
    results = results.assign(model_info=[model_info] * len(results))

    return results


def run_evaluation(workers=1, seed=42):

    print('Generating test data...')
    test_data = run_model_on_pendulum.generate_test_dataset(episodes=TEST_RUNS, seed=seed)
    total_runs = len(test_data)

    print('Running evaluation...')
    tick_t = timer()

    if workers > 1:
        workers = min(workers, total_runs)
        parallel = Parallel(n_jobs=workers, backend='loky', verbose=5)
        #runs_per_worker = np.ones(workers).astype(np.int) * (total_runs // workers)
        #runs_per_worker[-1] += total_runs % workers
        split_test_data = np.array_split(test_data, workers)

        for i in range(len(split_test_data)):
            print(len(split_test_data[i]))

        results = parallel(delayed(_evaluation_worker)(split_test_data[i]) for i in range(workers))
        results = pd.concat(results)
    else:
        results = _evaluation_worker(test_data)

    tock_t = timer()
    print(f'Done. Took ~{round(tock_t - tick_t)}s')
    print(f'Mean reward = {results.mean()}')

    results.to_csv(SAVE_PATH)


if __name__ == '__main__':
    run_evaluation(workers=7)
