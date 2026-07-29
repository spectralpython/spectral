'''
Tests for dimensionality reduction algorithms.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_dimensionality.py
'''

import numpy as np
import pytest

import spectral as spy


@pytest.fixture
def av3c_data(av3c_image):
    return av3c_image.load()


class TestDimensionality:
    '''Tests various math functions.'''

    def test_mnf_all_equals_data(self, av3c_data):
        '''Test that MNF transform with all components equals original data.'''
        data = av3c_data
        signal = spy.calc_stats(data)
        noise = spy.noise_from_diffs(data[117: 137, 85: 122, :])
        mnfr = spy.mnf(signal, noise)
        denoised = mnfr.denoise(data, num=data.shape[-1])
        assert (np.allclose(denoised, data))

    def test_ppi(self, av3c_data):
        '''Tests that ppi function runs'''
        data = av3c_data
        spy.ppi(data, 4)

    def test_ppi_threshold(self, av3c_data):
        '''Tests that ppi function runs with threshold arg'''
        data = av3c_data
        spy.ppi(data, 4, 10)

    def test_ppi_continues(self, av3c_data):
        '''Tests that running ppi with initial indices works as expected.'''
        data = av3c_data
        s = np.random.get_state()
        p = spy.ppi(data, 4)
        np.random.set_state(s)
        p2 = spy.ppi(data, 2)
        p2 = spy.ppi(data, 2, start=p2)
        assert (np.all(p == p2))

    def test_ppi_centered(self, av3c_data):
        '''Tests that ppi with mean-subtracted data works as expected.'''
        data = av3c_data
        s = np.random.get_state()
        p = spy.ppi(data, 4)

        np.random.set_state(s)
        data_centered = data - spy.calc_stats(data).mean
        p2 = spy.ppi(data_centered, 4)
        assert (np.all(p == p2))

    def test_smacc_minimal(self):
        '''Tests smacc correctness on minimal example.'''
        H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0]
        ])
        S, F, R = spy.smacc(H)
        assert (np.allclose(np.matmul(F, S) + R, H))
        assert (np.min(F) == 0.0)
        expected_S = np.array([
            # First two longer ones.
            [1., 1., 0.],
            [0., 1., 1.],
            # First of the two shorted ones. Other can be expressed other 3.
            [1., 0., 0.],
        ])
        assert (np.array_equal(S, expected_S))

    def test_smacc_runs(self, av3c_data):
        '''Tests that smacc runs without additional arguments.'''
        # Without scaling numeric errors accumulate.
        scaled_data = av3c_data / 10000
        S, F, R = spy.smacc(scaled_data)
        data_shape = scaled_data.shape
        H = scaled_data.reshape(data_shape[0] * data_shape[1], data_shape[2])
        assert (np.allclose(np.matmul(F, S) + R, H))
        assert (np.min(F) == 0.0)
        assert (len(S.shape) == 2 and S.shape[0] == 9 and S.shape[1] == 220)

    def test_smacc_min_endmembers(self, av3c_data):
        '''Tests that smacc runs with min_endmember argument.'''
        # Without scaling numeric errors accumulate.
        scaled_data = av3c_data / 10000
        S, F, R = spy.smacc(scaled_data, 10)
        data_shape = scaled_data.shape
        H = scaled_data.reshape(data_shape[0] * data_shape[1], data_shape[2])
        assert (np.allclose(np.matmul(F, S) + R, H))
        assert (np.min(F) == 0.0)
        assert (len(S.shape) == 2 and S.shape[0] == 10 and S.shape[1] == 220)

    def test_smacc_max_residual_norm(self, av3c_data):
        '''Tests that smacc runs with max_residual_norm argument.'''
        # Without scaling numeric errors accumulate.
        scaled_data = av3c_data / 10000
        S, F, R = spy.smacc(scaled_data, 9, 0.8)
        data_shape = scaled_data.shape
        H = scaled_data.reshape(data_shape[0] * data_shape[1], data_shape[2])
        assert (np.allclose(np.matmul(F, S) + R, H))
        assert (np.min(F) == 0.0)
        residual_norms = np.einsum('ij,ij->i', R, R)
        assert (np.max(residual_norms) <= 0.8)

    def test_pca_runs(self, av3c_data):
        '''Should be able to compute PCs and transform data.'''
        data = av3c_data
        spy.principal_components(data).transform(data)

    def test_pca_runs_from_stats(self, av3c_data):
        '''Should be able to pass image stats to PCA function.'''
        data = av3c_data
        stats = spy.calc_stats(data)
        spy.principal_components(stats).transform(data)

    def test_orthogonalize(self):
        '''Can correctly create an orthogonal basis from vectors.'''
        x = np.linspace(0, np.pi, 1001)
        # Create sin and cos vectors of unit length
        sin_h = np.sin(x)
        sin_h /= np.linalg.norm(sin_h)
        cos_h = np.cos(x)
        cos_h /= np.linalg.norm(cos_h)

        X = np.array([50 * sin_h, 75 * cos_h])
        Y = spy.orthogonalize(X)
        assert (np.allclose(Y.dot(Y.T), np.array([[1, 0], [0, 1]])))
        assert (np.allclose(X.dot(Y.T), np.array([[50, 0], [0, 75]])))

    def test_orthogonalize_subset(self):
        '''Can correctly create an orthogonal basis from vector subset.'''
        x = np.linspace(0, np.pi, 1001)
        # Create sin and cos vectors of unit length
        sin_h = np.sin(x)
        sin_h /= np.linalg.norm(sin_h)
        cos_h = np.cos(x)
        cos_h /= np.linalg.norm(cos_h)

        # First vector in X will already be a unit vector
        X = np.array([sin_h, 75 * cos_h])
        Y = spy.orthogonalize(X, start=1)
        assert (np.allclose(Y.dot(Y.T), np.array([[1, 0], [0, 1]])))
        assert (np.allclose(X.dot(Y.T), np.array([[1, 0], [0, 75]])))
