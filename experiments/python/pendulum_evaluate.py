from pathlib import Path

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

MODEL_PATH = "../out/tmp_mdn_model.pkl"
SAVE_PATH = "../out/tmp_mdn_model.csv"


if __name__ == '__main__':

    # init environment
    env = gym.make('GaussianPendulum-v0')

    # load model
    mdn = MDN_Model.load(Path(MODEL_PATH))
    mpc = MPC(env, mdn, horizon=20, n_action_sequences=2000, np_random=None)

    def next_mdn_mpc_action(obs):
        return mpc.get_action(obs)

    result_df = pd.DataFrame(columns=("mass", "mass_mean", "mass_std", "reward"))
    for mean, std in zip(MASS_MEANS, MASS_STDS):
        masses, rewards = run_model_on_pendulum.test(env, next_mdn_mpc_action, TEST_RUNS, mean, std)
        t, = masses.shape
        new_df = pd.DataFrame({"mass": masses, "mass_mean": [mean] * t, "mass_std": [std] * t, "reward": rewards})
        result_df = result_df.append(new_df)

    result_df.to_csv(SAVE_PATH)
