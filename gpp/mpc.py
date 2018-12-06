import gym
from gym.spaces import Discrete, Box
import numpy as np

from .reward_functions import RewardFunction
from .models import BaseModel


class MPC:

    def __init__(self, env: gym.Env, model: BaseModel, horizon: int, n_action_sequences: int,
                 np_random=None, reward_function=None, use_history=False):

        self.reward_function = reward_function or RewardFunction(env)

        if np_random is None:
            np_random = np.random.RandomState()
        self.np_random = np_random

        self.env = env
        self.n_action_sequences = n_action_sequences
        self.horizon = horizon
        self.model = model
        self.use_history = use_history
        self.a_history = []
        self.s_history = []

    def forget_history(self):
        self.a_history = []
        self.s_history = []

    @property
    def history(self):
        return self.s_history, self.a_history

    def get_action(self, current_state: np.ndarray):

        npr = self.np_random
        action_space = self.env.action_space
        current_state = current_state.copy()

        if isinstance(action_space, Box):
            all_actions = npr.uniform(action_space.low, action_space.high,
                                      (self.n_action_sequences, self.horizon, action_space.shape[0]))
        elif isinstance(action_space, Discrete):
            all_actions = npr.randint(0, action_space.n,
                                      (self.n_action_sequences, self.horizon))
        else:
            raise NotImplementedError

        if self.use_history:
            all_states = self.model.forward_sim(all_actions, current_state, history=self.history)
            self.s_history.append(current_state)
        else:
            all_states = self.model.forward_sim(all_actions, current_state)

        if hasattr(self.env.unwrapped, 'goal'):
            goal = self.env.unwrapped.goal
        else:
            goal = None

        if self.reward_function.use_dones:
            dones = np.zeros(self.n_action_sequences, dtype=np.bool)
        else:
            dones = None

        rewards = np.zeros(self.n_action_sequences)

        # for each timestep
        for t in range(self.horizon):
            rewards += self.reward_function(all_states[:, t], goal=goal, actions=all_actions[:, t], dones=dones)

        max_reward_i = rewards.argmax()
        print("Max reward: ", rewards[max_reward_i])
        print("Init reward: ", self.horizon * self.reward_function(current_state.reshape(1, -1), goal=goal, actions=np.zeros((1, 4))))
        best_action = all_actions[max_reward_i, 0]

        if self.use_history:
            self.a_history.append(best_action.copy())

        return best_action
