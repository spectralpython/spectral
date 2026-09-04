'''Tests of core functions/classes in `spectral.algorithms.algorithms`.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_algorithms.py
'''

import os
import pickle
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
import pytest

import spectral as spy
from spectral.algorithms.algorithms import (mean_cov, calc_stats, iterator,
                                            iterator_ij, ImageIterator,
                                            ImageMaskIterator, GaussianStats,
                                            ndvi, bdist, bdist_terms,
                                            cov_avg, covariance, log_det,
                                            transform_image, unmix,
                                            spectral_angles, msam,
                                            TrainingClass, TrainingClassSet,
                                            SampleIterator,
                                            create_training_classes, ppi)
from spectral.algorithms.transforms import LinearTransform
from spectral.io.spyfile import TransformedImage
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


class TestNdvi:

    @pytest.fixture
    def wide_data(self):
        '''A synthetic image with more bands than the shared `data`
        fixture, wide enough for multi-band red/nir ranges.'''
        return np.random.RandomState(0).rand(6, 5, 10)

    def test_single_band(self, wide_data):
        red = 2
        nir = 6
        result = ndvi(wide_data, red, nir)
        r = wide_data[:, :, red].astype(float)
        n = wide_data[:, :, nir].astype(float)
        assert_allclose(result, (n - r) / (n + r))

    def test_multi_band_range_is_unbiased_mean(self, wide_data):
        '''Regression test: `ndvi` used to sum (rather than average) a
        multi-band red/nir range using Python's builtin `sum()`, which
        doesn't operate along the band axis at all -- it silently
        produced a wrong-shaped, wrong-valued result. Using unevenly
        sized red/nir ranges here would also catch a reintroduced version
        of that bug that summed instead of averaged (which would bias the
        result toward whichever range has more bands).'''
        red = slice(2, 4)      # 2 bands
        nir = slice(5, 8)      # 3 bands
        result = ndvi(wide_data, red, nir)
        assert result.shape == wide_data.shape[:2]
        r = np.mean(wide_data[:, :, red].astype(float), axis=2)
        n = np.mean(wide_data[:, :, nir].astype(float), axis=2)
        assert_allclose(result, (n - r) / (n + r))


class TestBdist:

    @pytest.fixture
    def two_classes(self):
        rng = np.random.RandomState(0)
        d1 = rng.rand(50, 3)
        d2 = rng.rand(50, 3) + 2.0

        def make_class(d):
            stats = GaussianStats(mean=d.mean(0),
                                  cov=np.cov(d, rowvar=False),
                                  nsamples=d.shape[0])
            return SimpleNamespace(stats=stats)

        return (make_class(d1), make_class(d2))

    def test_bdist_is_sum_of_terms(self, two_classes):
        (c1, c2) = two_classes
        terms = bdist_terms(c1, c2)
        assert_allclose(bdist(c1, c2), terms[0] + terms[1])

    def test_bdist_terms_lin_term_zero_for_identical_classes(self,
                                                             two_classes):
        (c1, _) = two_classes
        (lin_term, quad_term) = bdist_terms(c1, c1)
        assert_allclose(lin_term, 0.0, atol=1e-10)
        assert_allclose(quad_term, 0.0, atol=1e-10)

    def test_bdist_positive_for_different_classes(self, two_classes):
        (c1, c2) = two_classes
        assert bdist(c1, c2) > 0


class TestCovAvg:

    @pytest.fixture
    def masked_image(self):
        rng = np.random.RandomState(0)
        img = rng.rand(10, 10, 4)
        mask = np.zeros((10, 10), int)
        mask[:5, :] = 1
        mask[5:, :] = 2
        return (img, mask)

    def test_weighted_matches_manual_calc(self, masked_image):
        (img, mask) = masked_image
        result = cov_avg(img, mask, weighted=True)
        s1 = calc_stats(img, mask, 1)
        s2 = calc_stats(img, mask, 2)
        N = s1.nsamples + s2.nsamples
        expected = (((s1.nsamples - 1) / float(N - 1)) * s1.cov +
                   ((s2.nsamples - 1) / float(N - 1)) * s2.cov)
        assert_allclose(result, expected)

    def test_unweighted_matches_manual_calc(self, masked_image):
        (img, mask) = masked_image
        result = cov_avg(img, mask, weighted=False)
        s1 = calc_stats(img, mask, 1)
        s2 = calc_stats(img, mask, 2)
        assert_allclose(result, (s1.cov + s2.cov) / 2)


class TestCovariance:

    def test_matches_mean_cov(self, data):
        assert_allclose(covariance(data), mean_cov(data)[1])


class TestLogDet:

    def test_matches_numpy_for_positive_definite(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        assert_allclose(log_det(A), np.log(np.linalg.det(A)))


class TestTransformImageAlgorithms:
    '''Tests `spectral.algorithms.algorithms.transform_image`, a distinct
    function from (but similar in purpose to) `spyfile.transform_image`.
    '''

    def test_ndarray_input(self, data):
        matrix = np.eye(data.shape[-1]) * 2
        result = transform_image(matrix, data)
        assert_allclose(result, data * 2)

    def test_spyfile_input_returns_transformed_image(self, av3c_image):
        matrix = np.eye(av3c_image.nbands)
        result = transform_image(matrix, av3c_image)
        assert isinstance(result, TransformedImage)

    def test_unrecognized_type_raises(self, data):
        matrix = np.eye(data.shape[-1])
        with pytest.raises(TypeError):
            transform_image(matrix, 'not an image')


class TestUnmix:

    def test_recovers_pure_and_mixed_pixels(self):
        endmembers = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        data = np.zeros((1, 3, 3))
        data[0, 0] = endmembers[0]
        data[0, 1] = endmembers[1]
        data[0, 2] = 0.5 * endmembers[0] + 0.5 * endmembers[1]
        result = unmix(data, endmembers)
        assert_allclose(result[0, 0], [1.0, 0.0], atol=1e-10)
        assert_allclose(result[0, 1], [0.0, 1.0], atol=1e-10)
        assert_allclose(result[0, 2], [0.5, 0.5], atol=1e-10)

    def test_dimension_mismatch_raises(self):
        endmembers = np.zeros((2, 3))
        data = np.zeros((1, 1, 5))
        with pytest.raises(AssertionError):
            unmix(data, endmembers)


class TestSpectralAngles:

    def test_same_direction_is_zero(self):
        data = np.zeros((1, 2, 3))
        data[0, 0] = [2.0, 0.0, 0.0]    # same direction as member, scaled
        data[0, 1] = [0.0, 0.0, 3.0]    # orthogonal to member
        members = np.array([[1.0, 0.0, 0.0]])
        angles = spectral_angles(data, members)
        assert_allclose(angles[0, 0, 0], 0.0, atol=1e-10)
        assert_allclose(angles[0, 1, 0], np.pi / 2)

    def test_dimension_mismatch_raises(self):
        members = np.zeros((2, 3))
        data = np.zeros((1, 1, 5))
        with pytest.raises(AssertionError):
            spectral_angles(data, members)


class TestMsam:

    def test_self_similarity_is_one(self):
        data = np.zeros((1, 1, 4))
        data[0, 0] = [1.0, 5.0, 2.0, 8.0]
        member = np.array([[1.0, 5.0, 2.0, 8.0]])
        result = msam(data, member)
        assert_allclose(result[0, 0, 0], 1.0)

    def test_matches_manual_formula(self):
        '''`msam` mean-centers and normalizes both the pixel and member
        spectra (a "Fisher z"-like transform) before computing the angle,
        so it isn't literally the same as the raw spectral angle -- this
        replicates its documented formula directly.'''
        rng = np.random.RandomState(0)
        pixel = rng.rand(5)
        member = rng.rand(5)
        data = pixel.reshape(1, 1, 5)
        members = member.reshape(1, 5)

        v = pixel - np.mean(pixel)
        v = v / np.sqrt(v.dot(v))
        m = member - np.mean(member)
        m = m / np.sqrt(m.dot(m))
        expected = 1.0 - np.arccos(np.clip(v.dot(m), -1, 1)) / (np.pi / 2)

        result = msam(data, members)
        assert_allclose(result[0, 0, 0], expected)

    def test_dimension_mismatch_raises(self):
        members = np.zeros((2, 3))
        data = np.zeros((1, 1, 5))
        with pytest.raises(AssertionError):
            msam(data, members)


@pytest.fixture
def masked_classes_image():
    '''A synthetic 6x6x4 image with two labeled classes in `mask`.'''
    img = np.random.RandomState(0).rand(6, 6, 4)
    mask = np.zeros((6, 6), int)
    mask[0:2, 0:2] = 1   # 4 pixels
    mask[3:5, 3:5] = 2   # 4 pixels
    return (img, mask)


class TestTrainingClass:

    def test_nbands_set_from_image_at_construction(self, masked_classes_image):
        '''Regression test: `nbands` used to always be reset to `None`
        right after construction (an unconditional assignment overwrote
        it), even when `image` was given.'''
        (img, mask) = masked_classes_image
        cl = TrainingClass(img, mask, index=1)
        assert cl.nbands == img.shape[2]

    def test_nbands_none_without_image(self, masked_classes_image):
        (_, mask) = masked_classes_image
        cl = TrainingClass(None, mask, index=1)
        assert cl.nbands is None

    def test_size_before_and_after_stats(self, masked_classes_image):
        (img, mask) = masked_classes_image
        cl = TrainingClass(img, mask, index=1)
        assert cl.size() == 4
        assert cl.stats_valid() is False
        cl.calc_stats()
        assert cl.stats_valid() is True
        assert cl.size() == 4

    def test_size_without_index_counts_all_nonzero(self, masked_classes_image):
        (img, mask) = masked_classes_image
        cl = TrainingClass(img, mask)  # index defaults to 0
        assert cl.size() == 8  # both labeled classes combined

    def test_iter_yields_all_class_pixels(self, masked_classes_image):
        (img, mask) = masked_classes_image
        cl = TrainingClass(img, mask, index=1)
        samples = list(cl)
        assert len(samples) == 4
        for s in samples:
            assert len(s) == img.shape[2]

    def test_transform(self, masked_classes_image):
        (img, mask) = masked_classes_image
        cl = TrainingClass(img, mask, index=1)
        cl.calc_stats()
        mean_before = cl.stats.mean.copy()
        cov_before = cl.stats.cov.copy()
        xform = np.eye(img.shape[2]) * 2
        cl.transform(xform)
        assert_allclose(cl.stats.mean, mean_before * 2)
        assert_allclose(cl.stats.cov, cov_before * 4)
        assert cl.nbands == img.shape[2]


class TestSampleIteratorAndAllSamples:

    def test_all_samples_returns_sample_iterator(self, masked_classes_image):
        (img, mask) = masked_classes_image
        classes = TrainingClassSet()
        classes.add_class(TrainingClass(img, mask, index=1))
        classes.add_class(TrainingClass(img, mask, index=2))
        it = classes.all_samples()
        assert isinstance(it, SampleIterator)
        samples = list(it)
        assert len(samples) == 8


class TestTrainingClassSet:

    @pytest.fixture
    def classes(self, masked_classes_image):
        (img, mask) = masked_classes_image
        classes = TrainingClassSet()
        classes.add_class(TrainingClass(img, mask, index=1))
        classes.add_class(TrainingClass(img, mask, index=2))
        return classes

    def test_getitem_and_len(self, classes):
        assert len(classes) == 2
        assert classes[1].index == 1
        assert classes[2].index == 2

    def test_add_class_duplicate_index_raises(self, classes,
                                              masked_classes_image):
        (img, mask) = masked_classes_image
        with pytest.raises(Exception):
            classes.add_class(TrainingClass(img, mask, index=1))

    def test_iter_yields_all_classes(self, classes):
        assert sorted(cl.index for cl in classes) == [1, 2]

    def test_calc_stats_computes_for_all_classes(self, classes):
        assert all(not cl.stats_valid() for cl in classes)
        classes.calc_stats()
        assert all(cl.stats_valid() for cl in classes)
        assert classes.nbands == 4

    def test_transform(self, classes):
        classes.calc_stats()
        xform = np.eye(4) * 2
        classes.transform(xform)
        assert classes.nbands == 4
        for cl in classes:
            assert cl.nbands == 4

    def test_save_raises_without_stats(self, classes, testdir):
        fname = os.path.join(testdir, 'no_stats.classes')
        with pytest.raises(Exception):
            classes.save(fname)

    def test_save_with_calc_stats_true_computes_and_saves(self, classes,
                                                          testdir):
        fname = os.path.join(testdir, 'auto_stats.classes')
        assert all(not cl.stats_valid() for cl in classes)
        classes.save(fname, calc_stats=True)
        assert all(cl.stats_valid() for cl in classes)
        assert os.path.isfile(fname)

    def test_save_and_load_round_trip(self, classes, testdir,
                                      masked_classes_image):
        (img, _) = masked_classes_image
        fname = os.path.join(testdir, 'roundtrip.classes')
        classes.calc_stats()
        classes.save(fname)

        loaded = TrainingClassSet()
        loaded.load(fname, img)
        assert len(loaded) == len(classes)
        for i in (1, 2):
            assert_allclose(loaded[i].stats.mean, classes[i].stats.mean)
            assert_allclose(loaded[i].stats.cov, classes[i].stats.cov)
            assert loaded[i].stats.nsamples == classes[i].stats.nsamples

    def test_load_raises_for_legacy_pickle_format(self, classes, testdir,
                                                  masked_classes_image):
        '''Files written by the old pickle-based `save` are no longer
        readable via `load` directly -- it should raise an explanatory
        exception (mentioning the conversion utility) rather than
        unpickling the file.'''
        (img, _) = masked_classes_image
        classes.calc_stats()
        fname = os.path.join(testdir, 'legacy.classes')
        ids = sorted(classes.classes.keys())
        with open(fname, 'wb') as f:
            pickle.dump(classes.classes[ids[0]].mask, f)
            pickle.dump(len(classes), f)
            for id in ids:
                c = classes.classes[id]
                pickle.dump(c.index, f)
                pickle.dump(c.stats.cov, f)
                pickle.dump(c.stats.mean, f)
                pickle.dump(c.stats.nsamples, f)
                pickle.dump(c.class_prob, f)

        loaded = TrainingClassSet()
        with pytest.raises(Exception, match='convert_legacy_class_file'):
            loaded.load(fname, img)

    def test_load_raises_for_non_npz_file(self, testdir,
                                          masked_classes_image):
        '''`load` should reject any file that isn't a valid `.npz`
        archive, without attempting to unpickle it.'''
        (img, _) = masked_classes_image
        fname = os.path.join(testdir, 'not_a_training_set.classes')
        with open(fname, 'wb') as f:
            f.write(b'not an npz file')

        loaded = TrainingClassSet()
        with pytest.raises(Exception, match='not a valid TrainingClassSet'):
            loaded.load(fname, img)


class TestCreateTrainingClassesIndices:

    def test_indices_kwarg_restricts_classes(self, masked_classes_image):
        (img, mask) = masked_classes_image
        classes = create_training_classes(img, mask, indices=[2])
        assert list(classes.classes.keys()) == [2]

    def test_default_uses_all_nonzero_mask_values(self, masked_classes_image):
        (img, mask) = masked_classes_image
        classes = create_training_classes(img, mask)
        assert sorted(classes.classes.keys()) == [1, 2]


class TestPpi:

    @pytest.fixture
    def small_image(self):
        return np.random.RandomState(0).rand(5, 5, 4)

    @pytest.mark.parametrize('bad_display', [-1, 1.5, True])
    def test_invalid_display_raises(self, small_image, bad_display):
        with pytest.raises(ValueError):
            ppi(small_image, niters=1, centered=True, display=bad_display)

    def test_display_calls_imshow_and_set_data(self, small_image,
                                               monkeypatch):
        set_data_calls = []
        titles = []

        class FakeFig:
            def set_data(self, data, **kwargs):
                set_data_calls.append(data.copy())

            def set_title(self, title):
                titles.append(title)

        fake_fig = FakeFig()
        imshow_calls = []

        def fake_imshow(data, **kwargs):
            imshow_calls.append(data.copy())
            return fake_fig

        monkeypatch.setattr(spy, 'imshow', fake_imshow)
        ppi(small_image, niters=6, centered=True, display=2)

        assert len(imshow_calls) == 1
        assert len(set_data_calls) == 2
        assert titles == ['PPI (2 iterations)', 'PPI (4 iterations)',
                          'PPI (6 iterations)']

    def test_keyboard_interrupt_not_updating_returns_counts(self,
                                                             small_image,
                                                             monkeypatch):
        '''An interrupt outside the narrow window where `counts` is being
        incremented should return the (uncorrupted) counts so far.'''
        def raise_interrupt(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(np.random, 'rand', raise_interrupt)
        result = ppi(small_image, niters=5, centered=True)
        assert result.shape == small_image.shape[:2]

    def test_keyboard_interrupt_during_update_returns_none(self,
                                                           small_image,
                                                           monkeypatch):
        '''An interrupt while `counts` is actually being incremented
        should return None, since the array may be left in a corrupted
        (partially-updated) state.'''
        class RaisingCounts(np.ndarray):
            def __setitem__(self, key, value):
                raise KeyboardInterrupt

        real_zeros = np.zeros

        def fake_zeros(*args, **kwargs):
            return real_zeros(*args, **kwargs).view(RaisingCounts)

        monkeypatch.setattr(np, 'zeros', fake_zeros)
        result = ppi(small_image, niters=5, centered=True)
        assert result is None
