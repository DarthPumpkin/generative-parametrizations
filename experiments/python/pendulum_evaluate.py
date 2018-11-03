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
MASS_MEANS = [1]
MASS_STDS = [0.5]
MODEL_TYPE = 'mdn'

if MODEL_TYPE == 'trpo':
    MODEL_PATH = "../out/tmp_trpo_model.pkl"
    SAVE_PATH = "../out/tmp_trpo_model.csv"
elif MODEL_TYPE == 'mdn':
    MODEL_PATH = "../out/tmp_mdn_model.pkl"
    SAVE_PATH = "../out/tmp_mdn_model.csv"


def _evaluation_worker(seed, runs):

    model_type = MODEL_TYPE

    # init environment
    env = gym.make('GaussianPendulum-v0')

    if model_type == 'mdn':

        # load model
        mdn = MDN_Model.load(Path(MODEL_PATH))
        mpc = MPC(env, mdn, horizon=20, n_action_sequences=2000, np_random=None)

        def next_action(obs):
            return mpc.get_action(obs)

    elif model_type == 'trpo':

        # load model
        model = TRPO.load(MODEL_PATH, env=env)

        def next_action(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action

    else:
        raise NotImplementedError

    results = pd.DataFrame()
    for mean, std in zip(MASS_MEANS, MASS_STDS):
        masses, rewards = run_model_on_pendulum.test(env, next_action, runs, mean, std, seed=seed)
        t, = masses.shape
        df = pd.DataFrame({"mass": masses, "mass_mean": [mean] * t, "mass_std": [std] * t, "reward": rewards})
        results = results.append(df)

    return results


def run_evaluation(workers=1, seed=42):

    tick_t = timer()
    print('Running evaluation...')

    if workers > 1:
        parallel = Parallel(n_jobs=workers, backend='multiprocessing', verbose=5)
        runs_per_worker = np.ones(workers).astype(np.int) * (TEST_RUNS // workers)
        runs_per_worker[-1] += TEST_RUNS % workers
        results = parallel(delayed(_evaluation_worker)(seed + i, runs_per_worker[i]) for i in range(workers))
        results = pd.concat(results)
    else:
        results = _evaluation_worker(seed, TEST_RUNS)

    tock_t = timer()
    print(f'Done. Took ~{round(tock_t - tick_t)}s')
    print(f'Mean reward = {results.mean()}')

    results.to_csv(SAVE_PATH)


if __name__ == '__main__':
    run_evaluation(workers=7)
