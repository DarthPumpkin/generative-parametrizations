import numpy as np
import torch
from torch import nn

HIDDEN_UNITS = 20
_normalization_factor = float(1.0 / np.sqrt(2.0 * np.pi))


class MDN(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_components):
        super(MDN, self).__init__()
        self.n_inputs = n_inputs
        self.n_components = n_components
        self.h1 = nn.Sequential(
            nn.Linear(n_inputs, HIDDEN_UNITS),
            nn.Tanh()
        )
        self.out_pi = nn.Sequential(
            nn.Linear(HIDDEN_UNITS, (n_outputs, n_components)),
            nn.Softmax(dim=-1)
        )
        self.out_mu = nn.Linear(HIDDEN_UNITS, (n_outputs, n_components))
        self.out_sig2 = nn.Linear(HIDDEN_UNITS, (n_outputs, n_components))

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """x: tensor (batch_size x n_inputs)
        :return: tensors pi, mu and sigma squared, each of size (batch_size x n_outputs x n_components)"""
        h1 = self.h1(x)
        pi = self.out_pi(h1)
        mu = self.out_mu(h1)
        sig2 = torch.exp(self.out_sig2(h1))
        return pi, mu, sig2


def mdn_loss(pi, mu, sig2, y):
    likelihoods = _gmm_neg_log_likelihood(pi, mu, sig2, y)
    return torch.mean(likelihoods)


def sample_gmm(pi: np.ndarray, mu: np.ndarray, sig2: np.ndarray) -> np.ndarray:
    """Takes one sample for each of the n GMMs
    :param pi: (n x k) mixture weights
    :param mu: (n x k) means
    :param sig2: (n x k) variances
    :return: (n) samples"""
    other_dims, k = pi.shape[:-1], pi.shape[-1]
    n = np.prod(other_dims)
    pi, mu, sig2 = [x.reshape(n, k) for x in (pi, mu, sig2)]
    k_sampled = _gumbel_sample(pi)
    mu_sampled = mu[np.arange(n), k_sampled]
    sig2_sampled = sig2[np.arange(n), k_sampled]
    gmm_sampled = np.random.randn(n) * sig2_sampled + mu_sampled
    return gmm_sampled.reshape(*other_dims)


def _gmm_neg_log_likelihood(pi, mu, sig2, y):
    """pi: 1xK, mu: 1xK, sig2: 1xK, y: Nx1"""
    exponent = -0.5 * (y - mu) ** 2 / sig2  # NxK
    densities = _normalization_factor * (torch.exp(exponent) / torch.sqrt(sig2))
    density = torch.sum(pi * densities, dim=-1)
    nnl = -torch.log(density)
    return nnl


def _gumbel_sample(x, axis=1):
    z = np.random.gumbel(loc=0, scale=1, size=x.shape)
    return (np.log(x) + z).argmax(axis=axis)

# def test_():
#     pi = torch.Tensor([[1]])
#     mu = torch.Tensor([[0]])
#     sig2 = torch.Tensor([[1]])
#     y = torch.Tensor([[-1],
#                       [0],
#                       [1]])
#     actual = _gmm_neg_log_likelihood(pi, mu, sig2, y)
#     expected = [0.5, 0, 0.5]
#     torch.testing.assert_allclose(actual, expected)
#    pi = torch.Tensor([0.5, 0.5])
#    mu = torch.Tensor([0, 0])
#    sig2 = torch.Tensor([1, 1])
#    y = torch.Tensor([-1, 0, 1])
#    actual = _gmm_neg_log_likelihood(pi, mu, sig2, y)
#    expected = [0.5, 0, 0.5]
#    torch.testing.assert_allclose(actual, expected)
# test_()

# def test_():
#     pi = torch.Tensor([0.5, 0.5])
#     mu = torch.Tensor([0, 0])
#     sig2 = torch.Tensor([1, 1])
#     y = torch.Tensor([-1, 0, 1])
#     actual = _gmm_neg_log_likelihood(pi, mu, sig2, y)
#     expected = -(np.log(0.5))
#     torch.testing.assert_allclose(actual, expected)
# test_()
