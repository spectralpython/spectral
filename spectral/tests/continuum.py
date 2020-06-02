'''
Runs unit tests for continuum processing functions.

To run the unit tests, type the following from the system command line:

    # python -m spectral.tests.continuum
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy.testing import assert_allclose

import spectral as spy
from spectral.algorithms.spymath import matrix_sqrt
from spectral.algorithms.continuum import spectral_continuum, remove_continuum, continuum_points
from spectral.tests.spytest import SpyTest


class ContinuumTest(SpyTest):
    def setup(self):
        self.image = spy.open_image('92AV3C.lan')
        self.bands = np.sort(
            spy.aviris.read_aviris_bands('92AV3C.spc').centers)


class FindContinuumTest(ContinuumTest):
    '''Tests spectral_continuum.'''

    def test_few_simple_cases(self):
        spectrum = np.array([1., 2., 2.5, 1.6, 0.75, 1.5, 2.2, 2.9, 1.8])
        bands = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
        expected = np.array([1., 2., 2.5, 2.58, 2.66, 2.74, 2.82, 2.9, 1.8])
        assert_allclose(expected, spectral_continuum(spectrum, bands))

        spectrum = np.array([0.6, 1., 2.45, 3.1, 3.25, 4.15,
                             4.35, 4.1, 3.1, 2.7, 2., 2.85, 3.75, 3., 2., 0.9])
        bands = np.array([0.3, 1., 1.8, 3., 4.5, 5.2, 6.45,
                          7., 7.1, 8., 8.1, 9., 9.3, 10.2, 10.5, 10.6])
        expected = np.array([0.6, 1.46333333, 2.45, 3.1, 3.81590909, 4.15, 4.35, 4.23421053,
                             4.21315789, 4.02368421, 4.00263158, 3.81315789, 3.75, 3., 2., 0.9])
        assert_allclose(expected, spectral_continuum(spectrum, bands))

        spectrum = np.array([0.5, 1.1, 1.5, 2.4, 1.9, 1.0])
        bands = np.array([0.5, 1.0, 1.7, 2.0, 3.0, 3.5])
        expected = np.array([0.5, 1.13333333, 2.02, 2.4, 1.9, 1.])
        assert_allclose(expected, spectral_continuum(spectrum, bands))

        spectrum = np.array([0.5, 1.1, 1.8, 2.0, 1.1, 0.9, 0.4])
        bands = np.array([0.5, 0.9, 1.6, 2.0, 2.1, 2.8, 3.0])
        expected = np.array([0.5, 1.1, 1.8, 2., 1.8625, 0.9, 0.4])
        assert_allclose(expected, spectral_continuum(spectrum, bands))

    def test_2d_array(self):
        part = self.image[20:22, 20:22].reshape(4, 220)
        cnt = spectral_continuum(part, self.bands)
        # Check some values to make sure results are sane.
        assert(cnt[0, 200] == 1422)
        assert(cnt[1, 200] == 1421)
        assert(cnt[2, 200] == 1469)
        assert(cnt[3, 200] == 1491)

    def test_3d_array(self):
        part = self.image[20:22, 20:22]
        cnt = spectral_continuum(part, self.bands)
        # Check some values to make sure results are sane.
        assert(cnt[0, 0, 200] == 1422)
        assert(cnt[0, 1, 200] == 1421)
        assert(cnt[1, 0, 200] == 1469)
        assert(cnt[1, 1, 200] == 1491)

    def test_out_parameter(self):
        part = self.image[20:22, 20:22]
        out = np.empty_like(part)
        cnt = spectral_continuum(part, self.bands, out=out)
        assert(cnt is out)
        # And just do a quick check if result is sane.
        assert(out[1, 1, 200] == 1491)


class FindContinuumPointsTest(ContinuumTest):
    '''Tests continuum_points.'''

    def test_points_of_real_spectrum(self):
        points = continuum_points(self.image[20, 20], self.bands)
        assert(np.array_equal(points[0], self.bands[[0, 1, 2, 5, 6, 41, 219]]))
        assert(np.array_equal(points[1], np.array(
            [3505, 4141, 4516, 4924, 5002, 4712, 1019], dtype=np.int16)))

class RemoveContinuumTest(ContinuumTest):
    '''Tests remove_continuum.'''

    def test_simple_case(self):
        continuum_removed = np.array([1., 0.6833713, 1., 1., 0.85169744,
                                      1., 1., 0.96830329, 0.73579013, 0.67102681,
                                      0.49967127, 0.74741201, 1., 1., 1., 1.])
        bands = np.array([0.30, 1.00, 1.80, 3.00, 4.50, 5.20, 6.45, 7.00, 7.10, 8.00,
                          8.10, 9.00, 9.30, 10.20, 10.50, 10.6])
        spectrum = np.array([0.60, 1.00, 2.45, 3.10, 3.25, 4.15, 4.35, 4.10, 3.10,
                             2.70, 2.00, 2.85, 3.75, 3.00, 2.00, 0.90])
        assert_allclose(continuum_removed, remove_continuum(spectrum, bands))

    def test_in_and_out_same(self):
        part = self.image[20:22, 20:22].astype(np.float64)
        res = remove_continuum(part, self.bands, out=part)
        # Make sure results are sane.
        assert(res[1, 1, 200] == 0.8372113957762342)
        assert(res is part)


def run():
    print('\n' + '-' * 72)
    print('Running continuum tests.')
    print('-' * 72)
    for T in [FindContinuumTest, FindContinuumPointsTest, RemoveContinuumTest]:
        T().run()


if __name__ == '__main__':
    from spectral.tests.run import parse_args, reset_stats, print_summary
    parse_args()
    reset_stats()
    run()
    print_summary()
