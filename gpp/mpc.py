import gym
import numpy as np

from .reward_functions import RewardFunction
from .models import BaseModel


class MPC:

    def __init__(self, env: gym.Env, model: BaseModel, horizon: int, n_action_sequences: int, np_random=None):

        self.reward_function = RewardFunction(env)

        if np_random is None:
            np_random = np.random.RandomState()
        self.np_random = np_random

        self.env = env
        self.n_action_sequences = n_action_sequences
        self.horizon = horizon
        self.model = model

    def get_action(self):

        npr = self.np_random
        action_space = self.env.action_space

        if hasattr(self.env.unwrapped, 'goal'):
            goal = self.env.unwrapped.goal
        else:
            goal = None

        all_actions = npr.uniform(action_space.low, action_space.high,
                                  (self.n_action_sequences, self.horizon, action_space.shape[0]))

        all_states = self.model.forward_sim(all_actions, self.env)

        rewards = np.zeros(all_actions.shape[0])

        # for each timestep
        for t in range(all_states.shape[1]):
            rewards += self.reward_function(all_states[:, t], goal=goal, actions=all_actions[:, t])

        max_reward_i = np.argmax(rewards)
        best_action = all_actions[max_reward_i, 0]

        return best_action
