import gym
from gym.spaces import Box
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch

from .models import BaseModel, VaeTorchModel, ComboTorchModel


class MpcBySgd:

    def __init__(self, env: gym.Env, model: BaseModel, horizon: int, n_action_sequences: int,
                 device=torch.device('cpu')):

        if not isinstance(model, VaeTorchModel) or not isinstance(model.env_model, ComboTorchModel):
            raise NotImplementedError

        self.env = env
        self.n_action_sequences = n_action_sequences
        self.horizon = horizon
        self.model = model
        self.device = device
        self.sgd_iters = 100
        self.sgd_learning_rate = 0.1
        self.a_history = []
        self.s_history = []

    def forget_history(self):
        self.a_history = []
        self.s_history = []

    def _init_optimizer(self, input_: torch.Tensor):
        optimizer = torch.optim.Adam([input_.requires_grad_()], lr=self.sgd_learning_rate)
        return optimizer

    @property
    def history(self):
        return self.s_history, self.a_history

    def get_action(self, current_state: np.ndarray, return_dream=False):

        action_space = self.env.action_space
        if not isinstance(action_space, Box):
            raise NotImplementedError

        current_state = current_state.copy()

        ################################################################################################################
        ################################################################################################################

        loss_fn = nn.MSELoss()
        loss_baseline = torch.zeros(1, requires_grad=False, device=self.device)

        action_seqs = torch.zeros((1, self.horizon, action_space.shape[0]), device=self.device)
        action_seqs = nn.init.kaiming_uniform_(action_seqs) # He init

        optimizer = self._init_optimizer(action_seqs)

        encoded_s = None
        dream = None

        for i in tqdm(range(self.sgd_iters)):

            action_seqs.data.clamp_(action_space.low.item(), action_space.high.item())

            optimizer.zero_grad()
            encoded_s, model_out = self.model.forward_sim(action_seqs, current_state, history=self.history,
                                                          return_dream=return_dream, mpc_by_sgd=True)

            if return_dream:
                model_out, dream = model_out

            if len(model_out.shape) > 1:
                rewards = model_out.sum(axis=1)
            else:
                rewards = model_out

            loss = loss_fn(rewards, loss_baseline)
            loss.backward()

            if i == 0 or i == self.sgd_iters-1:
                print('loss', i, loss.item())

            optimizer.step()

        action_seqs.data.clamp_(action_space.low.item(), action_space.high.item())

        ################################################################################################################
        ################################################################################################################

        self.s_history.append(encoded_s)

        best_action = action_seqs[0, 0].cpu().detach().numpy()
        self.a_history.append(best_action.copy())

        if return_dream:
            best_dream = dream[0]
            return best_action, best_dream
        else:
            return best_action
