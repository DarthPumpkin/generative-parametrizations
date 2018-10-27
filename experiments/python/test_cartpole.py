import gym
import numpy as np
from time import sleep

from gym.envs.classic_control import CartPoleEnv

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines import TRPO
from stable_baselines.common.policies import MlpPolicy


def init_env():
    """:rtype: (DummyVecEnv, CartPoleEnv)"""
    tm_env = gym.make('CartPole-v0')
    raw_env = tm_env.unwrapped
    monitor = Monitor(tm_env, None, allow_early_resets=True)
    env = DummyVecEnv([lambda: monitor])
    return env, raw_env


def visual_test(model_path: str):

    test_env, _ = init_env()

    model = TRPO.load(model_path, env=test_env)
    monitor = test_env.envs[0]  # type: Monitor
    assert isinstance(monitor, Monitor)

    raw_env = monitor.unwrapped  # type: CartPoleEnv
    assert isinstance(raw_env, CartPoleEnv)

    for _ in range(5):
        obs = test_env.reset()
        for _ in range(500):
            test_env.render()
            action, states = model.predict(obs)
            obs, rewards, dones, info = test_env.step(action)
            sleep(1./60)

    test_env.close()


def train(model_path: str):
    env, raw_env = init_env()
    raw_env.gravity = 98
    model = TRPO(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=300_000)
    model.save(model_path)


def main():
    model_path = '../out/tmp/cartpole_98_2.pkl'
    # train(model_path)
    visual_test(model_path)
    print('Done.')


if __name__ == '__main__':
    main()
