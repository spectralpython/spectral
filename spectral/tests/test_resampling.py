'''Tests of band-resampling functions and `BandResampler`.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_resampling.py
'''

import math

import numpy as np
from numpy.testing import assert_allclose
import pytest

from spectral import BandInfo
from spectral.algorithms.resampling import (erf_local, erfc, normal_cdf,
                                            normal_integral, ranges_overlap,
                                            overlap, normal, build_fwhm,
                                            create_resampling_matrix,
                                            BandResampler)


class TestErfLocal:
    '''`erf_local` is the Abramowitz & Stegun 7.1.26 approximation, used as
    a fallback when neither `math.erf` nor `scipy.special.erf` is
    available (never the case under any supported Python 3 -- this tests
    the approximation directly rather than through resampling.py's
    import-fallback chain, which is unreachable dead code today).'''

    @pytest.mark.parametrize('x', [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    def test_matches_math_erf(self, x):
        assert_allclose(erf_local(x), math.erf(x), atol=1e-6)


class TestErfc:

    def test_erfc_zero(self):
        assert_allclose(erfc(0.0), 1.0)

    def test_erfc_symmetry(self):
        assert_allclose(erfc(1.0) + erfc(-1.0), 2.0)


class TestNormalCdf:

    def test_cdf_at_mean_is_half(self):
        assert_allclose(normal_cdf(0.0), 0.5)

    def test_cdf_monotonic(self):
        assert normal_cdf(-1.0) < normal_cdf(0.0) < normal_cdf(1.0)


class TestNormalIntegral:

    def test_full_range_is_one(self):
        assert_allclose(normal_integral(-8, 8), 1.0, atol=1e-9)

    def test_zero_width_is_zero(self):
        assert normal_integral(2.0, 2.0) == 0.0

    def test_symmetric_range(self):
        assert_allclose(normal_integral(-1, 1), 2 * (normal_cdf(1) - 0.5))


class TestRangesOverlap:

    def test_overlapping(self):
        assert ranges_overlap([0, 2], [1, 3])

    def test_touching_boundary_counts_as_overlap(self):
        assert ranges_overlap([0, 1], [1, 2])

    def test_gap_no_overlap(self):
        assert not ranges_overlap([0, 1], [2, 3])

    def test_gap_no_overlap_reversed(self):
        assert not ranges_overlap([2, 3], [0, 1])


class TestOverlap:

    def test_overlap_bounds(self):
        assert overlap([0, 2], [1, 3]) == (1, 2)


class TestNormal:

    def test_peak_at_mean(self):
        assert normal(0.0, 1.0, 0.0) > normal(0.0, 1.0, 1.0)

    def test_symmetric_about_mean(self):
        assert_allclose(normal(5.0, 2.0, 3.0), normal(5.0, 2.0, 7.0))


class TestBuildFwhm:

    def test_endpoints_use_single_neighbor(self):
        fwhm = build_fwhm([1, 2, 4, 8])
        assert fwhm[0] == 1     # centers[1] - centers[0]
        assert fwhm[-1] == 4    # centers[-1] - centers[-2]

    def test_interior_uses_midpoint_difference(self):
        fwhm = build_fwhm([1, 2, 4, 8])
        assert fwhm[1] == 1.5   # (centers[2] - centers[0]) / 2
        assert fwhm[2] == 3.0   # (centers[3] - centers[1]) / 2


class TestCreateResamplingMatrix:

    def test_single_full_overlap_has_unit_weight(self):
        M = create_resampling_matrix([1, 2, 3], [1, 1, 1], [2], [1])
        assert_allclose(M, [[0.0, 1.0, 0.0]])

    def test_no_overlap_beyond_all_source_bands_is_nan(self):
        '''Target band entirely past the last source band: the search for
        an overlapping source band runs off the end of the source band
        list (`j == N1`) before ever finding a candidate.'''
        M = create_resampling_matrix([1, 2, 3], [1, 1, 1], [10], [1])
        assert np.isnan(M[0, 0])
        assert_allclose(M[0, 1:], [0.0, 0.0])

    def test_no_overlap_in_gap_between_source_bands_is_nan(self):
        '''Target band falls within a gap between two (narrow) source
        bands: a candidate source band is found before running off the
        end of the list, but it doesn't actually overlap.'''
        M = create_resampling_matrix([1, 2, 3], [0.1, 0.1, 0.1], [1.5],
                                     [0.1])
        assert np.isnan(M[0, 0])

    def test_symmetric_multi_band_overlap_splits_evenly(self):
        M = create_resampling_matrix([0, 2], [2, 2], [1], [1])
        assert_allclose(M, [[0.5, 0.5]])

    def test_weights_are_normalized(self):
        M = create_resampling_matrix([0, 1, 2, 3], [1.2, 1.2, 1.2, 1.2],
                                     [1.5], [1.0])
        assert_allclose(M.sum(), 1.0)


class TestBandResampler:

    def test_call_applies_matrix(self):
        resampler = BandResampler([1, 2, 3], [2], [1, 1, 1], [1])
        result = resampler([10.0, 20.0, 30.0])
        assert_allclose(result, [20.0])

    def test_accepts_bandinfo_arguments(self):
        src = BandInfo()
        src.centers = [1, 2, 3]
        src.bandwidths = [1, 1, 1]
        dst = BandInfo()
        dst.centers = [2]
        dst.bandwidths = [1]
        resampler = BandResampler(src, dst)
        assert_allclose(resampler([10.0, 20.0, 30.0]), [20.0])

    def test_default_fwhm_uses_build_fwhm(self):
        resampler = BandResampler([1, 2, 3], [1.5, 2.5])
        expected = create_resampling_matrix(
            [1, 2, 3], build_fwhm([1, 2, 3]),
            [1.5, 2.5], build_fwhm([1.5, 2.5]))
        assert_allclose(resampler.matrix, expected)

    def test_nan_for_unresampled_band(self):
        resampler = BandResampler([1, 2, 3], [1, 2, 3, 10], [1, 1, 1],
                                  [1, 1, 1, 1])
        result = resampler([10.0, 20.0, 30.0])
        assert np.isnan(result[-1])
