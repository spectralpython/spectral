'''
Runs unit tests for various target detectors.

To run the unit tests, type the following from the system command line:

    # python -m spectral.tests.detectors
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import spectral as spy
from spectral.tests.spytest import SpyTest


class MatchedFilterTest(SpyTest):
    def setup(self):
        from spectral.algorithms.detectors import MatchedFilter
        self.data = spy.open_image('92AV3C.lan').load()
        self.background = spy.calc_stats(self.data)
        self.target_ij = [33, 87]
#        self.target = self.data[33, 87]
        (i, j) = self.target_ij
        self.mf = MatchedFilter(self.background, self.data[i, j])

    def test_mf_bg_eq_zero(self):
        '''Matched Filter response of background should be zero.'''
        (i, j) = self.target_ij
        np.testing.assert_approx_equal(self.mf(self.background.mean), 0)

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


class RXTest(SpyTest):
    def setup(self):
        self.data = spy.open_image('92AV3C.lan').load()
        self.background = spy.calc_stats(self.data)

    def test_rx_bg_eq_zero(self):
        from spectral.algorithms.detectors import rx
        stats = spy.calc_stats(self.data)
        np.testing.assert_approx_equal(rx(stats.mean, background=stats), 0)


class ACETest(SpyTest):
    def setup(self):
        self.data = spy.open_image('92AV3C.lan').load()
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


def run():
    print('\n' + '-' * 72)
    print('Running target detector tests.')
    print('-' * 72)
    for T in [MatchedFilterTest, RXTest, ACETest]:
        T().run()


if __name__ == '__main__':
    from spectral.tests.run import parse_args, reset_stats, print_summary
    parse_args()
    reset_stats()
    run()
    print_summary()
