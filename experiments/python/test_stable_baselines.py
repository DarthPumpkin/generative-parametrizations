import os
import io
import gym
import pickle
import numpy as np
import scipy.stats
import tensorflow as tf
import matplotlib.pyplot as plt
from time import sleep

from stable_baselines.ddpg.policies import MlpPolicy as DDPGMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import ActorCriticRLModel
from stable_baselines.bench import Monitor
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import DDPG, TRPO, PPO2
from stable_baselines.her import HER
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy

from gym.envs.classic_control.pendulum import GaussianPendulumEnv
from utilities import make_tf_histogram, tf_summary_from_plot_buf


SET_NAME = 'set5'
OUT_DIR = f'../out/{SET_NAME}/'
TEST_RESULTS_FILE = f'../results/{SET_NAME}_test_results.pkl'
OVERWRITE_EXISTING_DATA = False

TRAIN = False
PERIODIC_TEST = True
PERIODIC_TEST_INTERVAL = 300 # episodes
FINAL_TEST = False
VISUAL_TEST = True

ALG = 'trpo'
TRAINING_STEPS = 700_000
TEST_RUNS = 200
TRIALS_PER_EXPERIMENT = 10

EXPERIMENTS = [
    {
        'name': 'pendulum_fixed_1000g',
        'mass_mean': 1.0,
        'mass_stdev': 0.0,
        'embed_knowledge': False
    },
    {
        'name': 'pendulum_fixed_400g',
        'mass_mean': 0.400,
        'mass_stdev': 0.0,
        'embed_knowledge': False
    },
    {
        'name': 'gaussian_pendulum_blind_l_range',
        'mass_mean': (0.050, 1.200),
        'mass_stdev': (0.010, 0.150),
        'embed_knowledge': False
    },
    {
        'name': 'gaussian_pendulum_informed_l_range',
        'mass_mean': (0.050, 1.200),
        'mass_stdev': (0.010, 0.150),
        'embed_knowledge': True
    },
    {
        'name': 'pendulum_exact_informed_s_range',
        'mass_mean': (0.050, 0.400),
        'perfect_knowledge': True,
        'embed_knowledge': True
    },
    {
        'name': 'pendulum_exact_informed_l_range',
        'mass_mean': (0.050, 1.200),
        'perfect_knowledge': True,
        'embed_knowledge': True
    },
    {
        'name': 'pendulum_exact_informed_db_range',
        'mass_mean': [(0.050, 0.300), (0.800, 1.200)],
        'perfect_knowledge': True,
        'embed_knowledge': True
    }
]

# EXPERIMENTS = [EXPERIMENTS[3]]


def plot_rewards_by_mass(runs, as_buf=True):

    plt.figure()
    plt.ylim(-2030, 70)
    plt.xlim(-0.05, 2.05)
    plt.scatter(runs[:, 2], runs[:, 3], facecolors='none', edgecolors='b')

    if as_buf:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf
    else:
        plt.show()


def visual_test(model_path: str, exp_config: dict):

    test_env, _ = init_env(exp_config)

    if ALG == 'ddpg':
        model = DDPG.load(model_path, env=test_env)
    elif ALG == 'trpo':
        model = TRPO.load(model_path, env=test_env)
    elif ALG == 'ppo2':
        model = PPO2.load(model_path, env=test_env)
    elif ALG == 'her':
        # model = HER.load(model_path, env=test_env)
        raise NotImplemented()
    else:
        raise ValueError(f'Unknown algorithm "{ALG}"!')

    monitor = test_env.envs[0]  # type: Monitor
    assert isinstance(monitor, Monitor)

    raw_env = monitor.unwrapped  # type: GaussianPendulumEnv
    assert isinstance(raw_env, GaussianPendulumEnv)

    for _ in range(5):
        obs = test_env.reset()
        mass_distr_params = raw_env.mass_distr_params
        sampled_mass = raw_env.physical_props[1]
        print(f'==> distribution params: {mass_distr_params} (mean, stdev) | sampled mass: {sampled_mass}')
        for _ in range(200):
            test_env.render()
            action, states = model.predict(obs)
            obs, rewards, dones, info = test_env.step(action)
            sleep(1./60)

    test_env.close()


def test(model_path: str, exp_config: dict):

    test_env, _ = init_env(exp_config)

    if ALG == 'ddpg':
        model = DDPG.load(model_path, env=test_env)
    elif ALG == 'trpo':
        model = TRPO.load(model_path, env=test_env)
    elif ALG == 'ppo2':
        model = PPO2.load(model_path, env=test_env)
    elif ALG == 'her':
        # model = HER.load(model_path, env=test_env)
        raise NotImplemented()
    else:
        raise ValueError(f'Unknown algorithm "{ALG}"!')

    monitor = test_env.envs[0] # type: Monitor
    assert isinstance(monitor, Monitor)

    raw_env = monitor.unwrapped # type: GaussianPendulumEnv
    assert isinstance(raw_env, GaussianPendulumEnv)

    raw_env.configure(
        seed=42,
        mass_mean=(0.05, 1.5),
        mass_stdev=(0.01, 0.15),
        embed_knowledge=exp_config.get('embed_knowledge', False),
        perfect_knowledge=exp_config.get('perfect_knowledge', False),
        gym_env=test_env
    )

    runs = np.zeros((TEST_RUNS, 4))

    for test_ep in range(runs.shape[0]):

        obs = test_env.reset()
        mass_distr_params = raw_env.mass_distr_params.copy()
        sampled_mass = raw_env.physical_props[1]

        while True:
            action, states = model.predict(obs)
            obs, rewards, dones, info = test_env.step(action)
            rewards_by_episode = monitor.episode_rewards
            episode = len(rewards_by_episode)
            if episode != test_ep:
                break

        last_tot_reward = rewards_by_episode[-1]
        runs[test_ep, :] = mass_distr_params[0], mass_distr_params[1], sampled_mass, last_tot_reward

    avg_reward = runs[:, 3].mean()
    print(f'Avg. test reward: {avg_reward}\n')

    return runs


def train(model_path: str, tmp_model_path: str, monitor_path: str, tensorboard_path: str, exp_config: dict):

    env, raw_env = init_env(exp_config, monitor_path)

    if ALG == 'ddpg':
        n_actions = env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
        model = DDPG(DDPGMlpPolicy, env, verbose=1, action_noise=action_noise, tensorboard_log=tensorboard_path)
    elif ALG == 'trpo':
        model = TRPO(MlpPolicy, env, verbose=0, tensorboard_log=tensorboard_path)
    elif ALG == 'ppo2':
        model = PPO2(MlpLstmPolicy, env, verbose=1, nminibatches=1)
    elif ALG == 'her':
        # model = HER(MlpPolicy, env, verbose=1)
        raise NotImplemented()
    else:
        raise ValueError(f'Unknown algorithm "{ALG}""!')

    global last_tested_ep
    last_tested_ep = 0

    def callback(locals_, globals_):
        episode_ = locals_.get('episodes_so_far') + 1 # type: int
        global last_tested_ep
        if episode_ - last_tested_ep < PERIODIC_TEST_INTERVAL:
            return
        last_tested_ep = episode_
        print(f'Episode {episode_}, testing...')
        step_ = locals_.get('timesteps_so_far') # type: int
        writer = locals_.get('writer') # type: tf.summary.FileWriter
        model_ = locals_.get('self') # type: ActorCriticRLModel
        model_.save(tmp_model_path)
        results_ = test(tmp_model_path, exp_config)
        avg_reward_ = results_[:, 3].mean()

        mean_reward_summary = tf.Summary()
        mean_reward_summary.value.add(tag='test_mean_reward', simple_value=avg_reward_)
        writer.add_summary(mean_reward_summary, step_)

        # useless at the moment
        hist_summary = make_tf_histogram('test_mass_distrib', results_[:, 2], bins=50)
        writer.add_summary(hist_summary, step_)

        plot_buf = plot_rewards_by_mass(results_)
        img_summary = tf_summary_from_plot_buf(plot_buf, 'test_reward_by_mass')
        writer.add_summary(img_summary, step_)

    cb = callback if PERIODIC_TEST else None
    model.learn(total_timesteps=exp_config.get('total_timesteps', TRAINING_STEPS), callback=cb)
    model.save(model_path)


def init_env(config: dict, monitor_file: str=None):
    """:rtype: (DummyVecEnv, GaussianPendulumEnv)"""
    tm_env = gym.make('GaussianPendulum-v0')
    raw_env = tm_env.unwrapped # type: GaussianPendulumEnv
    raw_env.configure(**config, gym_env=tm_env)
    monitor = Monitor(tm_env, monitor_file, allow_early_resets=True)
    env = DummyVecEnv([lambda: monitor])
    return env, raw_env


def main():

    if os.path.exists(TEST_RESULTS_FILE) and not OVERWRITE_EXISTING_DATA:
        with open(TEST_RESULTS_FILE, 'rb') as f:
            results_dict = pickle.load(f)
            assert isinstance(results_dict, dict)
    else:
        results_dict = {}

    for i, exp in enumerate(EXPERIMENTS):

        exp_name = exp['name']
        tb_dir = f'{OUT_DIR}/tensorboard/{exp_name}/'
        monitor_dir = f'{OUT_DIR}/monitor/'
        model_dir = f'{OUT_DIR}/model/{exp_name}/'
        os.makedirs(tb_dir, exist_ok=True)
        os.makedirs(monitor_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        n_trials = exp.get('trials', TRIALS_PER_EXPERIMENT)
        exp_avg_reward = np.zeros(n_trials)
        all_rewards = np.zeros((n_trials, TEST_RUNS))

        for j in range(n_trials):

            trial_name = f'{exp_name}_{j}'
            model_file = f'{model_dir}/{trial_name}.pkl'
            tmp_model_file = f'{model_dir}/{trial_name}_tmp.pkl'
            monitor_file = f'{monitor_dir}/{trial_name}'

            skip = False
            if TRAIN and os.path.exists(model_file):
                msg = 'Overwriting...' if OVERWRITE_EXISTING_DATA else 'Skipping...'
                print(f'Model for {trial_name} already exists. {msg}')
                skip = not OVERWRITE_EXISTING_DATA

            if TRAIN and not skip:
                print(f'Training {trial_name}...')
                train(model_file, tmp_model_file, monitor_file, tb_dir, exp)

            if os.path.exists(tmp_model_file):
                os.remove(tmp_model_file)

            skip = False
            if FINAL_TEST and trial_name in results_dict.keys():
                msg = 'Overwriting...' if OVERWRITE_EXISTING_DATA else 'Skipping...'
                print(f'Test results for {trial_name} already exist. {msg}')
                skip = not OVERWRITE_EXISTING_DATA

            if FINAL_TEST and not skip:
                print(f'Testing {trial_name}...')
                results = test(model_file, exp)
                results_dict[trial_name] = results
                with open(TEST_RESULTS_FILE, 'wb') as f:
                    pickle.dump(results_dict, f)

            results = results_dict[trial_name]
            all_rewards[j] = results[:, 3]
            avg_reward = results[:, 3].mean()
            exp_avg_reward[j] = avg_reward

            if VISUAL_TEST:
                print(f'\nRunning visual test for {trial_name} (test mean reward: {avg_reward})...')
                visual_test(model_file, exp)

        if FINAL_TEST:
            sem = scipy.stats.sem(exp_avg_reward)
            mean = exp_avg_reward.mean()
            stdev = exp_avg_reward.std()
            print(f'Experiment test reward stats: mean={mean}, stdev={stdev}, sem={sem}')

            sem = scipy.stats.sem(all_rewards, axis=None)
            mean = all_rewards.mean()
            stdev = all_rewards.std()
            print(f'Experiment test reward stats*: mean={mean}, stdev={stdev}, sem={sem}')

            avg_reward_by_ep = all_rewards.mean(axis=0)
            assert avg_reward_by_ep.size == TEST_RUNS
            sem = scipy.stats.sem(avg_reward_by_ep, axis=None)
            mean = avg_reward_by_ep.mean()
            stdev = avg_reward_by_ep.std()
            print(f'Experiment test reward stats**: mean={mean}, stdev={stdev}, sem={sem}\n')


if __name__ == '__main__':
    main()
    print('Done.')
