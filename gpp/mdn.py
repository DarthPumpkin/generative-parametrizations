import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, Gumbel

_normalization_factor = float(1.0 / np.sqrt(2.0 * np.pi))
_std_normal = Normal(0.0, 1.0)
_gumbel = Gumbel(0.0, 1.0)


class MDN(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_components, hidden_units=None):
        super(MDN, self).__init__()
        hidden_units = list(hidden_units or [20])
        self.n_inputs = n_inputs
        self.n_components = n_components
        self.n_outputs = n_outputs

        sizes = [n_inputs] + hidden_units
        self.h = nn.Sequential(*[nn.Sequential(
            nn.Linear(sizes[i], sizes[i+1]),
            nn.Tanh()
        ) for i in range(len(sizes) - 1)])

        self.out_pi = nn.Sequential(
            nn.Linear(hidden_units[-1], n_outputs * n_components),
            nn.Softmax(dim=-1)
        )
        self.out_mu = nn.Linear(hidden_units[-1], n_outputs * n_components)
        self.out_sig2 = nn.Linear(hidden_units[-1], n_outputs * n_components)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """x: tensor (batch_size x n_inputs)
        :return: tensors pi, mu and sigma squared, each of size (batch_size x n_outputs x n_components)"""
        h = self.h(x)
        pi = self.out_pi(h)
        mu = self.out_mu(h)
        sig2 = torch.exp(self.out_sig2(h))
        shape = (x.shape[0], self.n_outputs, self.n_components)
        return tuple([z.reshape(shape) for z in (pi, mu, sig2)])


def mdn_loss(pi, mu, sig2, y):
    """pi, mu, sig2: (batch_size x n_outputs x n_components), y: (batch_size x n_outputs)
    return: scalar"""
    n, m, k = pi.shape
    pi, mu, sig2 = [z.reshape(n * m, k) for z in (pi, mu, sig2)]
    y = y.reshape(n * m, 1)
    likelihoods = _gmm_neg_log_likelihood(pi, mu, sig2, y)
    return torch.mean(likelihoods)


def sample_gmm_torch(pi: torch.Tensor, mu: torch.Tensor, sig2: torch.Tensor) -> torch.Tensor:
    """
    Takes one sample for each of the n GMMs
    :param pi: (n x k) mixture weights
    :param mu: (n x k) means
    :param sig2: (n x k) variances
    :return: (n) samples
    """
    other_dims, k = pi.shape[:-1], pi.shape[-1]
    n = np.prod(other_dims).item()
    pi, mu, sig2 = [x.reshape(n, k) for x in (pi, mu, sig2)]
    k_sampled = _gumbel_sample_torch(pi)
    mu_sampled = mu[np.arange(n), k_sampled]
    sig2_sampled = sig2[np.arange(n), k_sampled]
    gmm_sampled = _std_normal.sample((n,)).to(pi.device) * sig2_sampled + mu_sampled
    return gmm_sampled.reshape(other_dims)


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
    """pi: NxK, mu: NxK, sig2: NxK, y: Nx1
    return: N"""
    exponent = -0.5 * (y - mu) ** 2 / sig2  # NxK
    densities = _normalization_factor * (torch.exp(exponent) / torch.sqrt(sig2))
    density = torch.sum(pi * densities, dim=-1)

    # if the density is very small, the log is NaN. This is a workaround...
    density = torch.clamp(density, min=1e-20)
    nnl = -torch.log(density)
    return nnl


def _gumbel_sample_torch(x: torch.Tensor, dim=1) -> torch.Tensor:
    z = _gumbel.sample(x.shape).to(x.device)
    return (torch.log(x) + z).argmax(dim=dim)


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
