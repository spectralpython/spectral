'''Tests of core functions/classes in `spectral.algorithms.algorithms`.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_algorithms.py
'''

import numpy as np
from numpy.testing import assert_allclose
import pytest

import spectral as spy
from spectral.algorithms.algorithms import (mean_cov, calc_stats, iterator,
                                            iterator_ij, ImageIterator,
                                            ImageMaskIterator)
from spectral.algorithms.transforms import LinearTransform
from spectral.utilities.errors import NaNValueError


class NonNdarrayImage:
    '''Wraps an ndarray without subclassing it, so that `mean_cov`/
    `calc_stats`/`iterator` dispatch to their generic Iterator-based code
    path (the one used for SpyFile input) instead of the ndarray fast
    path. All existing tests exercise the two code paths -- of `mean_cov`
    only the ndarray fast path is exercised, since every existing caller
    passes an already-loaded ndarray.
    '''
    def __init__(self, data):
        self._data = data
        self.shape = data.shape
        self.dtype = data.dtype

    def __getitem__(self, key):
        return self._data[key]


@pytest.fixture
def data():
    '''Small, deterministic synthetic image (6x5 pixels, 4 bands).'''
    return np.random.RandomState(0).rand(6, 5, 4)


class TestMeanCov:
    '''Tests `mean_cov`'s various input-type code paths.'''

    def test_3d_ndarray(self, data):
        (mean, cov, n) = mean_cov(data)
        pixels = data.reshape(-1, data.shape[-1])
        assert_allclose(mean, pixels.mean(axis=0))
        assert_allclose(cov, np.cov(pixels, rowvar=False))
        assert n == pixels.shape[0]

    def test_2d_ndarray(self, data):
        pixels = data.reshape(-1, data.shape[-1])
        (mean, cov, n) = mean_cov(pixels)
        assert_allclose(mean, pixels.mean(axis=0))
        assert_allclose(cov, np.cov(pixels, rowvar=False))
        assert n == pixels.shape[0]

    def test_invalid_ndim_raises(self, data):
        with pytest.raises(ValueError):
            mean_cov(data[0, 0])  # 1D

    def test_ndarray_with_mask(self, data):
        mask = np.zeros(data.shape[:2], int)
        mask[1:3, 2:4] = 1
        (mean, cov, n) = mean_cov(data, mask)
        pixels = data[mask != 0]
        assert_allclose(mean, pixels.mean(axis=0))
        assert n == pixels.shape[0]

    def test_ndarray_with_mask_and_index(self, data):
        mask = np.zeros(data.shape[:2], int)
        mask[0, 0] = 1
        mask[2, 2] = 2
        mask[4, 4] = 2
        (mean, cov, n) = mean_cov(data, mask, index=2)
        pixels = data[mask == 2]
        assert_allclose(mean, pixels.mean(axis=0))
        assert n == 2

    def test_non_ndarray_matches_ndarray_fast_path(self, data):
        '''The generic Iterator-based code path (used for non-ndarray,
        e.g. SpyFile, input) should produce identical results to the
        ndarray fast path, for the same data.'''
        wrapped = NonNdarrayImage(data)
        (mean, cov, n) = mean_cov(wrapped)
        (mean_ref, cov_ref, n_ref) = mean_cov(data)
        assert_allclose(mean, mean_ref)
        assert_allclose(cov, cov_ref)
        assert n == n_ref

    def test_non_ndarray_with_mask_and_index(self, data):
        mask = np.zeros(data.shape[:2], int)
        mask[0, 0] = 1
        mask[2, 2] = 2
        mask[4, 4] = 2
        wrapped = NonNdarrayImage(data)
        (mean, cov, n) = mean_cov(wrapped, mask, index=2)
        (mean_ref, cov_ref, n_ref) = mean_cov(data, mask, index=2)
        assert_allclose(mean, mean_ref)
        assert_allclose(cov, cov_ref)
        assert n == n_ref

    def test_accepts_iterator_instance_directly(self, data):
        '''Passing an already-constructed Iterator (rather than an image)
        should use it as-is rather than wrapping it again.'''
        it = ImageIterator(NonNdarrayImage(data))
        (mean, cov, n) = mean_cov(it)
        (mean_ref, cov_ref, n_ref) = mean_cov(data)
        assert_allclose(mean, mean_ref)
        assert_allclose(cov, cov_ref)
        assert n == n_ref


class TestIteratorFunction:

    def test_returns_same_instance_for_iterator_input(self, data):
        it = ImageIterator(NonNdarrayImage(data))
        assert iterator(it) is it

    def test_returns_image_mask_iterator_for_mask(self, data):
        mask = np.ones(data.shape[:2], int)
        it = iterator(NonNdarrayImage(data), mask)
        assert isinstance(it, ImageMaskIterator)

    def test_returns_image_iterator_without_mask(self, data):
        it = iterator(NonNdarrayImage(data))
        assert isinstance(it, ImageIterator)


class TestIteratorIj:

    def test_invalid_mask_ndim_raises(self):
        with pytest.raises(ValueError):
            list(iterator_ij(np.zeros((3, 3, 3))))


class TestImageMaskIterator:

    def test_shape_mismatch_raises(self, data):
        bad_mask = np.zeros((data.shape[0] + 1, data.shape[1]))
        with pytest.raises(ValueError):
            ImageMaskIterator(NonNdarrayImage(data), bad_mask)


class TestCalcStats:

    def test_nan_raises_by_default(self, data):
        bad_data = data.copy()
        bad_data[0, 0, 0] = np.nan
        with pytest.raises(NaNValueError):
            calc_stats(bad_data)

    def test_allow_nan(self, data):
        bad_data = data.copy()
        bad_data[0, 0, 0] = np.nan
        stats = calc_stats(bad_data, allow_nan=True)
        assert np.isnan(stats.mean[0])


@pytest.fixture
def pc(data):
    return spy.principal_components(data)


class TestPrincipalComponentsReduce:

    def test_reduce_num(self, pc):
        reduced = pc.reduce(num=2)
        assert reduced.eigenvectors.shape[1] == 2
        assert_allclose(reduced.eigenvalues, pc.eigenvalues[:2])

    def test_reduce_eigs(self, pc):
        reduced = pc.reduce(eigs=[0, 2])
        assert_allclose(reduced.eigenvalues, pc.eigenvalues[[0, 2]])
        assert_allclose(reduced.eigenvectors, pc.eigenvectors[:, [0, 2]])

    def test_reduce_fraction_retains_enough_variance(self, pc):
        reduced = pc.reduce(fraction=0.5)
        total = pc.eigenvalues.sum()
        assert (reduced.eigenvalues.sum() / total).real >= 0.5
        assert len(reduced.eigenvalues) < len(pc.eigenvalues)

    def test_reduce_fraction_one_retains_all(self, pc):
        '''`fraction=1.0` should hit the "no reduction achieved" branch
        (the cumulative variance only reaches 100% at the very last
        eigenvalue), returning `self` unchanged.'''
        reduced = pc.reduce(fraction=1.0)
        assert reduced is pc

    def test_reduce_fraction_out_of_range_raises(self, pc):
        with pytest.raises(Exception):
            pc.reduce(fraction=1.5)
        with pytest.raises(Exception):
            pc.reduce(fraction=0)

    def test_reduce_no_kwarg_raises(self, pc):
        with pytest.raises(Exception):
            pc.reduce()


class TestPrincipalComponentsDenoise:

    def test_full_rank_reconstructs_data(self, pc, data):
        denoised = pc.denoise(data, num=data.shape[-1])
        assert_allclose(denoised, data, atol=1e-8)

    def test_get_denoising_transform_matches_denoise(self, pc, data):
        f = pc.get_denoising_transform(num=2)
        assert_allclose(f(data), pc.denoise(data, num=2))


class TestGaussianStatsTransform:

    def test_transform(self, data):
        stats = spy.calc_stats(data)
        xform = LinearTransform(np.eye(data.shape[-1]) * 2)
        transformed = stats.transform(xform)
        assert_allclose(transformed.mean, xform(stats.mean))
        assert_allclose(transformed.cov,
                        xform._A.dot(stats.cov).dot(xform._A.T))
        assert transformed.nsamples == stats.nsamples

    def test_transform_rejects_non_lineartransform(self, data):
        stats = spy.calc_stats(data)
        with pytest.raises(TypeError):
            stats.transform(np.eye(data.shape[-1]))


@pytest.fixture
def mnfr(data):
    signal = spy.calc_stats(data)
    noise = spy.noise_from_diffs(data)
    return spy.mnf(signal, noise)


class TestMNFResult:

    def test_denoise_full_components_reconstructs_data(self, mnfr, data):
        denoised = mnfr.denoise(data, num=data.shape[-1])
        assert_allclose(denoised, data, atol=1e-6)

    def test_reduce_num_shape(self, mnfr, data):
        reduced = mnfr.reduce(data, num=2)
        assert reduced.shape == data.shape[:2] + (2,)

    def test_num_with_snr_monotonic(self, mnfr):
        low = mnfr.num_with_snr(-0.99)
        high = mnfr.num_with_snr(1000)
        assert low >= high

    def test_reduce_by_snr_matches_equivalent_num(self, mnfr, data):
        n = mnfr.num_with_snr(0)
        reduced_by_snr = mnfr.reduce(data, snr=0)
        reduced_by_num = mnfr.reduce(data, num=int(n))
        assert_allclose(reduced_by_snr, reduced_by_num)

    def test_num_from_kwargs_requires_one_kwarg(self, mnfr):
        with pytest.raises(Exception):
            mnfr._num_from_kwargs()

    def test_num_from_kwargs_rejects_both(self, mnfr):
        with pytest.raises(Exception):
            mnfr._num_from_kwargs(num=2, snr=5)

    def test_num_from_kwargs_rejects_unknown_kwarg(self, mnfr):
        with pytest.raises(Exception):
            mnfr._num_from_kwargs(bogus=1)
