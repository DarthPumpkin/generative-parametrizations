import tempfile
import unittest
from pathlib import Path
from unittest import TestCase

import torch
import numpy as np

import _gpp
from gpp import mdn
from gpp.models import MDN_Model


class TestMDNModel(TestCase):

    def test_save_and_load(self):

        n_inputs = 10
        n_outputs = 11
        n_components = 12

        model = MDN_Model(n_inputs, n_outputs, n_components, device=torch.device('cuda'))
        self.assertTrue(model.device.type == 'cuda')
        tmp_path = Path(f'{tempfile.mkdtemp()}/test_model.pkl')
        model.save(tmp_path)

        del model

        model = MDN_Model.load(tmp_path)
        self.assertTrue(model.n_inputs == n_inputs)
        self.assertTrue(model.n_outputs == n_outputs)
        self.assertTrue(model.n_components == n_components)
        self.assertTrue(model.device.type == 'cuda')

        model.device = torch.device('cpu')
        self.assertTrue(model.device.type == 'cpu')

        tmp_path.unlink()
        tmp_path.parent.rmdir()


class Test_sample_gmm(TestCase):

    def test_sample_gmm(self):
        pi = np.ones((4, 3, 2)) * 0.5
        mu = np.linspace(-1, 1, num=4 * 3 * 2).reshape((4, 3, 2))
        sig2 = np.ones((4, 3, 2)) * 0.01
        expected = np.mean(mu, axis=-1)
        actual = np.mean([mdn.sample_gmm(pi, mu, sig2) for _ in range(10000)], axis=0)
        np.testing.assert_allclose(actual, expected, atol=0.01)

    def test_sample_gmm_torch(self):
        pi = torch.ones((4, 3, 2)) * 0.5
        mu = torch.linspace(-1, 1, 4 * 3 * 2).reshape((4, 3, 2))
        sig2 = torch.ones((4, 3, 2)) * 0.01
        expected = torch.mean(mu, dim=-1)
        foo = torch.Tensor([mdn.sample_gmm_torch(pi, mu, sig2).numpy() for _ in range(10000)])
        actual = torch.mean(foo, dim=0)
        torch.testing.assert_allclose(actual, expected, atol=0.01, rtol=1e-7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
