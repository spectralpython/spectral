'''Tests various target detectors.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_detectors.py
'''

import numpy as np
import pytest

import spectral as spy


class TestMatchedFilter:
    @pytest.fixture(autouse=True)
    def setup(self, av3c_image):
        from spectral.algorithms.detectors import MatchedFilter
        self.data = av3c_image.load()
        self.background = spy.calc_stats(self.data)
        self.target_ij = [33, 87]
#        self.target = self.data[33, 87]
        (i, j) = self.target_ij
        self.mf = MatchedFilter(self.background, self.data[i, j])

    def test_mf_bg_eq_zero(self):
        '''Matched Filter response of background should be zero.'''
        (i, j) = self.target_ij
        np.testing.assert_approx_equal(self.mf(self.background.mean).item(), 0)

    def test_mf_target_eq_one(self):
        '''Matched Filter response of target should be one.'''
        from spectral.algorithms.detectors import matched_filter
        (i, j) = self.target_ij
        target = self.data[i, j]
        mf = matched_filter(self.data, target, self.background)
        np.testing.assert_approx_equal(mf[i, j], 1)

    def test_mf_target_no_bg_eq_one(self):
        '''Matched Filter response of target should be one.'''
        from spectral.algorithms.detectors import matched_filter
        (i, j) = self.target_ij
        target = self.data[i, j]
        mf = matched_filter(self.data, target)
        np.testing.assert_approx_equal(mf[i, j], 1)

    def test_mf_target_pixel_eq_one(self):
        '''Matched Filter response of target pixel should be one.'''
        (i, j) = self.target_ij
        np.testing.assert_approx_equal(self.mf(self.data)[i, j], 1)

    def test_mf_windowed_target_eq_one(self):
        '''Windowed Matched Filter response of target pixel should be one.'''
        X = self.data[:10, :10, :]
        ij = (3, 3)
        y = spy.matched_filter(X, X[ij], window=(3, 7), cov=self.background.cov)
        np.allclose(1, y[ij])

    def test_mf_windowed_target_eq_one_no_cov(self):
        '''Windowed Matched Filter without a supplied `cov` (background
        covariance recomputed for every window) should also score the
        target pixel as one. A bigger window is needed here than in
        `test_mf_windowed_target_eq_one` since, without a precomputed
        `cov`, the outer window must contain at least as many pixels as
        there are bands (220) to estimate a full-rank covariance.'''
        X = self.data[:25, :25, :]
        ij = (12, 12)
        y = spy.matched_filter(X, X[ij], window=(3, 21))
        assert np.allclose(1, y[ij])

    def test_mf_whiten(self):
        '''`whiten` should match its documented closed-form definition.'''
        X = self.data[:5, :5, :].reshape(-1, self.data.shape[-1])
        whitened = self.mf.whiten(X)
        expected = (np.sqrt(self.mf.coef) * self.background.sqrt_inv_cov).dot(
            (X - self.background.mean).T).T
        assert np.allclose(whitened, expected)

    def test_matched_filter_mutually_exclusive_raises(self):
        from spectral.algorithms.detectors import matched_filter
        (i, j) = self.target_ij
        with pytest.raises(ValueError):
            matched_filter(self.data, self.data[i, j],
                           background=self.background, window=(3, 7))


class TestRX:
    @pytest.fixture(autouse=True)
    def setup(self, av3c_image):
        self.data = av3c_image.load()
        self.background = spy.calc_stats(self.data)

    def test_rx_bg_eq_zero(self):
        from spectral.algorithms.detectors import rx
        stats = spy.calc_stats(self.data)
        np.testing.assert_approx_equal(rx(stats.mean, background=stats), 0)

    def test_rx_call_without_background_computes_stats(self):
        from spectral.algorithms.detectors import RX
        detector = RX()
        assert detector.background is None
        scores = detector(self.data[:5, :5, :])
        assert detector.background is not None
        assert scores.shape == (5, 5)

    def test_rx_call_rejects_non_ndarray(self):
        from spectral.algorithms.detectors import RX
        with pytest.raises(TypeError):
            RX()([1, 2, 3])

    def test_rx_mutually_exclusive_raises(self):
        from spectral.algorithms.detectors import rx
        with pytest.raises(ValueError):
            rx(self.data, background=self.background, window=(3, 7))

    def test_rx_windowed_flags_outlier(self):
        '''A pixel that's a clear spectral outlier relative to its
        surrounding window should score much higher than typical
        (non-outlier) pixels.'''
        from spectral.algorithms.detectors import rx
        img = np.random.RandomState(0).normal(
            size=(20, 20, 5)).astype(float)
        outlier_ij = (10, 10)
        img[outlier_ij] += 20.0
        scores = rx(img, window=(3, 9))
        assert scores[outlier_ij] > 10 * np.median(scores)

    def test_rx_windowed_with_cov_flags_outlier(self):
        '''Same as `test_rx_windowed_flags_outlier`, but supplying a
        precomputed `cov` (only the background mean is recomputed per
        window).'''
        from spectral.algorithms.detectors import rx
        img = np.random.RandomState(0).normal(
            size=(20, 20, 5)).astype(float)
        outlier_ij = (10, 10)
        img[outlier_ij] += 20.0
        cov = spy.calc_stats(img).cov
        scores = rx(img, window=(3, 9), cov=cov)
        assert scores[outlier_ij] > 10 * np.median(scores)


class TestACE:
    @pytest.fixture(autouse=True)
    def setup(self, av3c_image):
        self.data = av3c_image.load()
        self.bg = spy.calc_stats(self.data)
        self.X = self.data[:20, :20, :]

    def test_ace_bg_eq_zero(self):
        '''ACE score of background mean should be zero.'''
        ij = (10, 10)
        y = spy.ace(self.bg.mean, self.X[ij], background=self.bg)
        assert (np.allclose(0, y))

    def test_ace_pixel_target_eq_one(self):
        '''ACE score of target should be one for single pixel arg.'''
        ij = (10, 10)
        y = spy.ace(self.X[ij], self.X[ij], background=self.bg)
        assert (np.allclose(1, y))

    def test_ace_novec_pixel_target_eq_one(self):
        '''ACE score of target should be one for single pixel arg.'''
        ij = (10, 10)
        y = spy.ace(self.X[ij], self.X[ij], background=self.bg, vectorize=False)
        assert (np.allclose(1, y))

    def test_ace_target_eq_one(self):
        '''ACE score of target should be one.'''
        ij = (10, 10)
        y = spy.ace(self.X, self.X[ij], background=self.bg)
        assert (np.allclose(1, y[ij]))

    def test_ace_novec_target_eq_one(self):
        '''ACE score (without vectorization) of target should be one.'''
        ij = (10, 10)
        y = spy.ace(self.X, self.X[ij], background=self.bg, vectorize=False)
        assert (np.allclose(1, y[ij]))

    def test_ace_multi_targets_eq_one(self):
        '''ACE score of multiple targets should each be one.'''
        ij1 = (10, 10)
        ij2 = (3, 12)
        y = spy.ace(self.X, [self.X[ij1], self.X[ij2]], background=self.bg)
        assert (np.allclose(1, [y[ij1][0], y[ij2][1]]))

    def test_ace_novec_multi_targets_eq_one(self):
        '''ACE score of multiple targets should each be one.'''
        ij1 = (10, 10)
        ij2 = (3, 12)
        y = spy.ace(self.X, [self.X[ij1], self.X[ij2]], background=self.bg,
                    vectorize=False)
        assert (np.allclose(1, [y[ij1][0], y[ij2][1]]))

    def test_ace_multi_targets_bg_eq_zero(self):
        '''ACE score of background for multiple targets should be one.'''
        ij1 = (10, 10)
        ij2 = (3, 12)
        y = spy.ace(self.bg.mean, [self.X[ij1], self.X[ij2]],
                    background=self.bg)
        assert (np.allclose(0, y))

    def test_ace_subspace_targets_eq_one(self):
        '''ACE score of targets defining target subspace should each be one.'''
        ij1 = (10, 10)
        ij2 = (3, 12)
        y = spy.ace(self.X, np.array([self.X[ij1], self.X[ij2]]),
                    background=self.bg)
        assert (np.allclose(1, [y[ij1], y[ij2]]))

    def test_ace_novec_subspace_targets_eq_one(self):
        '''ACE score of targets defining target subspace should each be one.'''
        ij1 = (10, 10)
        ij2 = (3, 12)
        y = spy.ace(self.X, np.array([self.X[ij1], self.X[ij2]]),
                    background=self.bg, vectorize=False)
        assert (np.allclose(1, [y[ij1], y[ij2]]))

    def test_ace_subspace_bg_eq_zero(self):
        '''ACE score of background for target subspace should be zero.'''
        ij1 = (10, 10)
        ij2 = (3, 12)
        y = spy.ace(self.bg.mean, np.array([self.X[ij1], self.X[ij2]]),
                    background=self.bg)
        assert (np.allclose(0, y))

    def test_ace_windowed_target_eq_one(self):
        '''ACE score of target for windowed background should be one.'''
        ij = (10, 10)
        y = spy.ace(self.X, self.X[ij], window=(3, 7), cov=self.bg.cov)
        assert (np.allclose(1, y[ij]))

    def test_ace_windowed_target_eq_one_no_cov(self):
        '''Same as `test_ace_windowed_target_eq_one`, but without a
        precomputed `cov` (background mean and covariance are both
        recomputed for every window). A bigger outer window is needed
        here since, without a precomputed `cov`, it must contain at least
        as many pixels as there are bands (220) to estimate a full-rank
        covariance.'''
        ij = (10, 10)
        y = spy.ace(self.X, self.X[ij], window=(3, 19))
        assert (np.allclose(1, y[ij]))

    def test_ace_windowed_multi_targets_eq_one(self):
        '''Windowed ACE score of each target, for a list of multiple
        targets, should be one at that target's own pixel.'''
        ij1 = (5, 5)
        ij2 = (12, 14)
        y = spy.ace(self.X, [self.X[ij1], self.X[ij2]], window=(3, 19))
        assert y.shape == self.X.shape[:2] + (2,)
        assert np.allclose(1, y[ij1][0])
        assert np.allclose(1, y[ij2][1])

    def test_ace_windowed_multi_targets_eq_one_with_cov(self):
        '''Same as `test_ace_windowed_multi_targets_eq_one`, but with a
        precomputed `cov`.'''
        ij1 = (5, 5)
        ij2 = (12, 14)
        y = spy.ace(self.X, [self.X[ij1], self.X[ij2]], window=(3, 7),
                    cov=self.bg.cov)
        assert y.shape == self.X.shape[:2] + (2,)
        assert np.allclose(1, y[ij1][0])
        assert np.allclose(1, y[ij2][1])

    def test_ace_set_target_none(self):
        from spectral.algorithms.detectors import ACE
        ij = (10, 10)
        detector = ACE(self.X[ij], background=self.bg)
        assert detector._P is not None
        detector.set_target(None)
        assert detector._target is None
        assert detector._P is None

    def test_ace_call_rejects_non_ndarray(self):
        from spectral.algorithms.detectors import ACE
        ij = (10, 10)
        detector = ACE(self.X[ij], background=self.bg)
        with pytest.raises(TypeError):
            detector([1, 2, 3])

    def test_ace_auto_background(self):
        '''Constructing an ACE detector without `background` and calling
        it directly should compute background stats from the given data.'''
        from spectral.algorithms.detectors import ACE
        ij = (10, 10)
        detector = ACE(self.X[ij])
        assert detector._background is None
        scores = detector(self.X)
        assert detector._background is not None
        assert np.allclose(1, scores[ij])

    def test_ace_function_auto_background_multi_target(self):
        '''`spy.ace` with a list of targets and no `background`/`window`
        should compute background stats once from the given data.'''
        ij1 = (5, 5)
        ij2 = (12, 14)
        y = spy.ace(self.X, [self.X[ij1], self.X[ij2]])
        assert np.allclose(1, y[ij1][0])
        assert np.allclose(1, y[ij2][1])

    def test_ace_function_mutually_exclusive_raises(self):
        ij = (10, 10)
        with pytest.raises(ValueError):
            spy.ace(self.X, self.X[ij], background=self.bg, window=(3, 7))
