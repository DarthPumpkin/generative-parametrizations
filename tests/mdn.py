import unittest
from unittest import TestCase

import numpy as np

from gpp import mdn


class Test_sample_gmm(TestCase):
    def test_sample_gmm(self):
        pi = np.ones((4, 3, 2)) * 0.5
        mu = np.linspace(-1, 1, num=4 * 3 * 2).reshape((4, 3, 2))
        sig2 = np.ones((4, 3, 2)) * 0.01
        expected = np.mean(mu, axis=-1)
        actual = np.mean([mdn.sample_gmm(pi, mu, sig2) for _ in range(10000)], axis=0)
        np.testing.assert_allclose(actual, expected, atol=0.01)


if __name__ == '__main__':
    unittest.main(verbosity=2)
