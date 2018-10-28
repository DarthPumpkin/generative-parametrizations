import gym
import numpy as np

from .reward_functions import RewardFunction
from .models.base import BaseModel


class MPC:

    def __init__(self, env: gym.Env, model: BaseModel, horizon: int, n_action_sequences: int, np_random=None):

        self.reward_function = RewardFunction(env)

        if np_random is None:
            np_random = np.random
        self.np_random = np_random

        self.env = env
        self.n_action_sequences = n_action_sequences
        self.horizon = horizon
        self.model = model

    def get_action(self):

        npr = self.np_random
        action_space = self.env.action_space
        goal = None

        all_samples = npr.uniform(action_space.low, action_space.high,
                                  (self.n_action_sequences, self.horizon, action_space.shape[0]))

        all_states = self.model.forward_sim(all_samples, self.env)

        rewards = np.zeros(all_samples.shape[0])

        # for each sequence of actions
        for s in range(all_states.shape[0]):

            # for each timestep
            for t in range(all_states.shape[1]):

                rewards[s] += self.reward_function(all_states[s, t], goal)

        max_reward_i = np.argmax(rewards)
        best_action = all_samples[max_reward_i, 0]

        return best_action
