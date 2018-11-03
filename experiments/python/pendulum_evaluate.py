from pathlib import Path

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
    # load model
    path = Path(MODEL_PATH)

    mdn = MDN_Model.load(path)
    mpc = MPC()

    def next_mdn_mpc_action(obs):
        return mdn.forward_sim(action.reshape(1, 1, -1), obs).reshape(-1)

    result_df = pd.DataFrame(columns=("mass", "mass_mean", "mass_std", "reward"))
    for mean, std in zip(MASS_MEANS, MASS_STDS):
        masses, rewards = run_model_on_pendulum.test(next_mdn_mpc_action, TEST_RUNS, mean, std)
        t, = masses.shape
        result_df.append({"mass": masses, "mass_mean": [mean] * t, "mass_std": [std] * t, "reward": rewards})

    result_df.to_csv(SAVE_PATH)
