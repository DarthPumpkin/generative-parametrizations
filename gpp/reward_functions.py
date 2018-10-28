import gym
from gym.envs.robotics import FetchEnv
import numpy as np


class RewardFunction:

    def __init__(self, env: gym.Env):

        unwrapped = env.unwrapped

        if isinstance(unwrapped, FetchEnv):
            self._reward_fn = _build_reward_fn_fetch_env(unwrapped)

        elif isinstance(unwrapped, gym.GoalEnv):
            self._reward_fn = _build_reward_fn_goal_env(unwrapped)

        else:
            raise NotImplemented(f'No reward function for environment of type {type(env)}')

    def __call__(self, x: np.ndarray, goal: np.ndarray):
        return self._reward_fn(x, goal)


def _build_reward_fn_fetch_env(env: FetchEnv):

    reward_type = env.reward_type
    distance_threshold = env.distance_threshold

    if env.has_object:
        # achieved goal is object_pos
        achieved_goal_idx = np.arange(3, 6)
    else:
        # achieved goal is gripper pos
        achieved_goal_idx = np.arange(0, 3)

    def reward_fn(x: np.ndarray, goal: np.ndarray):
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

    def reward_fn(x: np.ndarray, goal: np.ndarray):
        rewards = np.zeros(x.shape[0])
        # for each timestep
        for t in range(x.shape[0]):
            rewards[t] += compute_reward(x[t], goal, {})
        return rewards

    return reward_fn
