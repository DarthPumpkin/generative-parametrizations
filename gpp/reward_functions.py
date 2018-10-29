import gym
from gym.envs.robotics import FetchEnv
from gym.envs.classic_control import PendulumEnv
import numpy as np


class RewardFunction:

    def __init__(self, env: gym.Env):

        unwrapped = env.unwrapped

        if isinstance(unwrapped, FetchEnv):
            self._reward_fn = _build_reward_fn_fetch_env(unwrapped)

        elif isinstance(unwrapped, gym.GoalEnv):
            self._reward_fn = _build_reward_fn_goal_env(unwrapped)

        elif isinstance(unwrapped, PendulumEnv):
            self._reward_fn = _build_reward_fn_pendulum_env(unwrapped)

        else:
            raise NotImplemented(f'No reward function for environment of type {type(env)}')

    def __call__(self, states: np.ndarray, goal: np.ndarray=None, actions: np.ndarray=None):
        return self._reward_fn(states, goal=goal, actions=actions)


def _build_reward_fn_pendulum_env(env: PendulumEnv):

    max_torque = env.max_torque

    def angle_normalize(x):
        return np.fmod(x + np.pi, 2*np.pi) - np.pi

    def reward_fn(states: np.ndarray, actions: np.ndarray=None, **kwargs):
        actions = np.clip(actions, -max_torque, max_torque)
        x = states.take([0], axis=-1)
        y = states.take([1], axis=-1)
        thdot = states.take([2], axis=-1)
        th = np.arctan2(y, x)
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (actions ** 2)
        return -costs.ravel()

    return reward_fn


def _build_reward_fn_fetch_env(env: FetchEnv):

    reward_type = env.reward_type
    distance_threshold = env.distance_threshold

    if env.has_object:
        # achieved goal is object_pos
        achieved_goal_idx = np.arange(3, 6)
    else:
        # achieved goal is gripper pos
        achieved_goal_idx = np.arange(0, 3)

    def reward_fn(x: np.ndarray, goal: np.ndarray=None, **kwargs):
        x = x.take(achieved_goal_idx, axis=-1)
        d = np.linalg.norm(x - goal, axis=-1)
        if reward_type == 'sparse':
            return -(d > distance_threshold).astype(np.float32)
        else:
            return -d

    return reward_fn


def _build_reward_fn_goal_env(env: gym.GoalEnv):
    # TODO: Check shapes

    compute_reward = env.compute_reward

    def reward_fn(x: np.ndarray, goal: np.ndarray=None, **kwargs):
        rewards = np.zeros(x.shape[0])
        # for each timestep
        for t in range(x.shape[0]):
            rewards[t] += compute_reward(x[t], goal, {})
        return rewards

    return reward_fn
