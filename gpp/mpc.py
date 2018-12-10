import gym
from gym.spaces import Discrete, Box
import numpy as np

from .reward_functions import RewardFunction
from .models import BaseModel


class MPC:

    def __init__(self, env: gym.Env, model: BaseModel, horizon: int, n_action_sequences: int,
                 np_random=None, reward_function=None, use_history=False, direct_reward=False,
                 action_period=1):

        self.reward_function = reward_function or RewardFunction(env)

        if np_random is None:
            np_random = np.random.RandomState()
        self.np_random = np_random

        self.env = env
        self.n_action_sequences = n_action_sequences
        self.horizon = horizon
        self.model = model
        self.direct_reward = direct_reward
        self.use_history = use_history
        self.action_period = action_period
        self.a_history = []
        self.s_history = []

        if action_period > 1 and not use_history:
            raise ValueError

    def forget_history(self):
        self.a_history = []
        self.s_history = []

    @property
    def history(self):
        return self.s_history, self.a_history

    def get_action(self, current_state: np.ndarray, discretized_actions=None, return_dream=False):

        npr = self.np_random
        action_space = self.env.action_space
        current_state = current_state.copy()

        if isinstance(action_space, Box):
            if discretized_actions is not None:
                all_actions_idx = npr.randint(0, len(discretized_actions),
                                              (self.n_action_sequences, self.horizon))
                all_actions = discretized_actions[all_actions_idx].copy()
            else:
                all_actions = npr.uniform(action_space.low, action_space.high,
                                          (self.n_action_sequences, self.horizon, action_space.shape[0]))
        elif isinstance(action_space, Discrete):
            all_actions = npr.randint(0, action_space.n,
                                      (self.n_action_sequences, self.horizon))
        else:
            raise NotImplementedError

        if self.use_history:

            step = len(self.s_history)

            if step == 0 or step % self.action_period == 0:
                encoded_s, model_out = self.model.forward_sim(all_actions, current_state,
                                                              history=self.history, return_dream=return_dream)
                self.s_history.append(encoded_s)
            else:
                if return_dream:
                    raise NotImplementedError
                encoded_s = self.model.forward_sim(all_actions, current_state, encoding_only=True)
                prev_a = self.a_history[-1]
                self.s_history.append(encoded_s)
                self.a_history.append(prev_a)
                return prev_a
        else:
            model_out = self.model.forward_sim(all_actions, current_state, return_dream=return_dream)

        if hasattr(self.env.unwrapped, 'goal'):
            goal = self.env.unwrapped.goal
        else:
            goal = None

        if self.reward_function.use_dones:
            dones = np.zeros(self.n_action_sequences, dtype=np.bool)
        else:
            dones = None

        if return_dream:
            model_out, dream = model_out
        else:
            dream = None

        if self.direct_reward:
            if len(model_out.shape) > 1:
                rewards = model_out.sum(axis=1)
            else:
                rewards = model_out
        else:
            rewards = np.zeros(self.n_action_sequences)

            # for each timestep
            for t in range(self.horizon):
                rewards += self.reward_function(model_out[:, t], goal=goal, actions=all_actions[:, t], dones=dones)

        max_reward_i = rewards.argmax()
        best_action = all_actions[max_reward_i, 0]
        # print("Pred. reward: ", rewards[max_reward_i] / self.horizon)
        # print("Init reward: ", self.horizon * self.reward_function(current_state.reshape(1, -1), goal=goal, actions=np.zeros((1, 1))))

        if self.use_history:
            self.a_history.append(best_action.copy())

        if return_dream:
            best_dream = dream[max_reward_i]
            return best_action, best_dream
        else:
            return best_action
