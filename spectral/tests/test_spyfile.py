'''
Tests of spectral file I/O functions.

The unit tests in this module open the sample image "92AV3C.lan" (bundled
under `tests/data/`) and save it in various formats (different combinations
of byte order, interleave, and data type); for each file written, the new
file is opened and known data values are read and checked to verify they are
read properly. A second set of tests does the same for synthetically
generated complex-valued data.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_spyfile.py
'''

import functools
import os
import warnings

import numpy as np
import pytest

import spectral as spy

ENVI_COMPLEX_TEST_SIZES = [64, 28]

INTERLEAVES = ('bil', 'bip', 'bsq')
BYTEORDERS = ('big', 'little')

# Canonical dtype names, matching what `SpyFileTestSuite` used to compute
# via `np.dtype(d).name` for each abbreviated dtype code.
DTYPES = [np.dtype(d).name for d in ('i2', 'i4', 'f4', 'f8', 'c8', 'c16')]

REAL_FILE_DATUM = (99, 99, 99)
REAL_FILE_VALUE = 2057.0

COMPLEX_SHAPE = (100, 200, 64)
COMPLEX_DATUM = (33, 44, 25)

assert_almost_equal = np.testing.assert_allclose


def assert_allclose(a, b, **kwargs):
    np.testing.assert_allclose(np.array(a), np.array(b), **kwargs)


def assert_same_shape_almost_equal(obj1, obj2, decimal=7, err_msg='',
                                   verbose=True):
    """
    Assert that two objects are almost equal and have the same shape.

    numpy.testing.assert_almost_equal does test for shape, but considers
    arrays with one element and a scalar to be the same.
    """
    # Types might be different since ImageArray stores things as
    # floats by default.
    if np.isscalar(obj1):
        assert np.isscalar(obj2), err_msg
    else:
        assert obj1.shape == obj2.shape, err_msg

    assert_almost_equal(obj1, obj2, err_msg=err_msg, verbose=verbose)


def _complex_test_dtypes():
    '''Complex dtype names supported by both numpy and ENVI.'''
    dtypes = []
    for s in spy.COMPLEX_SIZES:
        if s not in ENVI_COMPLEX_TEST_SIZES:
            continue
        name = 'complex{}'.format(s)
        if hasattr(np, name):
            dtypes.append(name)
        else:
            # This is unlikely to happen because numpy currently supports
            # more complex types than ENVI.
            warnings.warn('numpy does not support {}. Skipping test.'.format(name))
    return dtypes


COMPLEX_DTYPES = _complex_test_dtypes()


@functools.lru_cache(maxsize=None)
def _complex_source(dtype):
    '''Returns `(array, datum, value)` of random complex data for `dtype`.

    The array is generated once per dtype (and cached) so that every
    interleave/byteorder variant written for that dtype is derived from the
    same source values.
    '''
    X = np.array(np.random.rand(*COMPLEX_SHAPE)
                + 1j * np.random.rand(*COMPLEX_SHAPE), dtype=dtype)
    return (X, COMPLEX_DATUM, X[COMPLEX_DATUM])


class _SpyFileReadTests:
    '''Tests that SpyFile methods read data correctly from files.

    Subclasses provide a `spyfile_case` fixture that yields
    `(image, datum, value)`, where `datum` is a 3-tuple `(i, j, k)` giving
    the row, column, and band of a datum to check, and `value` is the
    expected scalar value at that location. `i` and `j` should be at least
    10 pixels away from the edge of the associated image and `k` should
    have at least 10 bands above and below it in the image.
    '''

    def test_read_datum(self, spyfile_case):
        (image, datum, value) = spyfile_case
        assert_almost_equal(image.read_datum(*datum, use_memmap=True),
                            value)
        assert_almost_equal(image.read_datum(*datum, use_memmap=False),
                            value)

    def test_read_pixel(self, spyfile_case):
        (image, datum, value) = spyfile_case
        (i, j, k) = datum
        assert_almost_equal(image.read_pixel(i, j, use_memmap=True)[k],
                            value)
        assert_almost_equal(image.read_pixel(i, j, use_memmap=False)[k],
                            value)

    def test_read_band(self, spyfile_case):
        (image, datum, value) = spyfile_case
        (i, j, k) = datum
        assert_almost_equal(image.read_band(k, use_memmap=True)[i, j],
                            value)
        assert_almost_equal(image.read_band(k, use_memmap=False)[i, j],
                            value)

    def test_read_bands(self, spyfile_case):
        (image, datum, value) = spyfile_case
        (i, j, k) = datum
        bands = (k - 5, k - 2, k, k + 1)
        assert_almost_equal(image.read_bands(bands,
                                            use_memmap=True)[i, j, 2],
                            value)
        assert_almost_equal(image.read_bands(bands,
                                            use_memmap=False)[i, j, 2],
                            value)

    def test_read_bands_nonascending(self, spyfile_case):
        (image, datum, value) = spyfile_case
        (i, j, k) = datum
        bands = (k - 2, k + 1, k, k - 5)
        assert_almost_equal(image.read_bands(bands,
                                            use_memmap=True)[i, j, 2],
                            value)
        assert_almost_equal(image.read_bands(bands,
                                            use_memmap=False)[i, j, 2],
                            value)

    def test_read_bands_duplicates(self, spyfile_case):
        (image, datum, value) = spyfile_case
        (i, j, k) = datum
        bands = (k - 5, k - 5, k, k - 5)
        assert_almost_equal(image.read_bands(bands,
                                            use_memmap=True)[i, j, 2],
                            value)
        assert_almost_equal(image.read_bands(bands,
                                            use_memmap=False)[i, j, 2],
                            value)

    def test_read_subregion(self, spyfile_case):
        (image, datum, value) = spyfile_case
        (i, j, k) = datum
        region = image.read_subregion((i - 5, i + 9),
                                      (j - 3, j + 4), use_memmap=True)
        assert_almost_equal(region[5, 3, k], value)
        region = image.read_subregion((i - 5, i + 9),
                                      (j - 3, j + 4), use_memmap=False)
        assert_almost_equal(region[5, 3, k], value)

    def test_read_subimage(self, spyfile_case):
        (image, datum, value) = spyfile_case
        (i, j, k) = datum
        subimage = image.read_subimage([0, 3, i, 5],
                                       [1, j, 4, 7],
                                       [3, 7, k], use_memmap=True)
        assert_almost_equal(subimage[2, 1, 2], value)
        subimage = image.read_subimage([0, 3, i, 5],
                                       [1, j, 4, 7],
                                       [3, 7, k], use_memmap=False)
        assert_almost_equal(subimage[2, 1, 2], value)

        subimage = image.read_subimage([0, 3, i, 5],
                                       [1, j, 4, 7], use_memmap=True)
        assert_almost_equal(subimage[2, 1, k], value)
        subimage = image.read_subimage([0, 3, i, 5],
                                       [1, j, 4, 7], use_memmap=False)
        assert_almost_equal(subimage[2, 1, k], value)

    def test_load(self, spyfile_case):
        (image, datum, value) = spyfile_case
        (i, j, k) = datum
        data = image.load()
        spyf = image

        load_assert = assert_allclose
        load_assert(data[i, j, k], value)
        first_band = spyf[:, :, 0]
        load_assert(data[:, :, 0], first_band)
        # This is checking if different ImageArray and SpyFile indexing
        # results are the same shape, so we can't just reuse the already
        # loaded first band.
        load_assert(data[:, 0, 0].squeeze(), spyf[:, 0, 0].squeeze())
        load_assert(data[0, 0, 0], spyf[0, 0, 0])
        load_assert(data[0, 0], spyf[0, 0])
        load_assert(data[-1, -1, -1], spyf[-1, -1, -1])
        load_assert(data[-1, -3:-1], spyf[-1, -3:-1])
        load_assert(data[(6, 25)], spyf[(6, 25)])

        # The following test would currently fail, because
        # SpyFile.__get_item__ treats [6,25] the same as (6,25).

        # load_assert(data[[6, 25]],
        #             spyf[[6, 25]])

        load_assert(data.read_band(0), spyf.read_band(0))
        load_assert(data.read_bands([0, 1]), spyf.read_bands([0, 1]))
        load_assert(data.read_pixel(1, 2), spyf.read_pixel(1, 2))
        load_assert(data.read_subregion([0, 3], [1, 2]),
                    spyf.read_subregion([0, 3], [1, 2]))
        load_assert(data.read_subregion([0, 3], [1, 2], [0, 1]),
                    spyf.read_subregion([0, 3], [1, 2], [0, 1]))
        load_assert(data.read_subimage([0, 2, 4], [6, 3]),
                    spyf.read_subimage([0, 2, 4], [6, 3]))
        load_assert(data.read_subimage([0, 2], [6, 3], [0, 1]),
                    spyf.read_subimage([0, 2], [6, 3], [0, 1]))
        load_assert(data.read_datum(1, 2, 8), spyf.read_datum(1, 2, 8))

        ufunc_result = data + 1
        assert isinstance(ufunc_result, np.ndarray)
        assert not isinstance(ufunc_result, type(data))
        non_ufunc_result = data.diagonal()
        assert isinstance(non_ufunc_result, np.ndarray)
        assert not isinstance(non_ufunc_result, type(data))

    def test_getitem_i_j_k(self, spyfile_case):
        (image, datum, value) = spyfile_case
        (i, j, k) = datum
        assert_almost_equal(image[i, j, k], value)

    def test_getitem_i_j(self, spyfile_case):
        (image, datum, value) = spyfile_case
        (i, j, k) = datum
        assert_almost_equal(image[i, j][k], value)

    def test_getitem_i_j_kslice(self, spyfile_case):
        (image, datum, value) = spyfile_case
        (i, j, k) = datum
        assert_almost_equal(image[i, j, k-2:k+3:2][0, 0, 1], value)

    def test_getitem_islice_jslice(self, spyfile_case):
        (image, datum, value) = spyfile_case
        (i, j, k) = datum
        assert_almost_equal(image[i-3:i+3, j-3:j+3][3, 3, k], value)


@pytest.fixture(scope='class', params=INTERLEAVES)
def interleave(request):
    return request.param


@pytest.fixture(scope='class', params=BYTEORDERS)
def byteorder(request):
    return request.param


@pytest.fixture(scope='class', params=[True, False], ids=['memmap', 'direct'])
def use_memmap(request):
    return request.param


class TestSpyFile(_SpyFileReadTests):
    '''Tests reading "92AV3C.lan" after saving it with every combination of
    interleave, dtype, and byte order, with and without memmap access
    (where the combination supports memmap).

    `interleave`, `byteorder`, and `use_memmap` are class-scoped
    parametrized fixtures defined above (shared with `TestSpyFileComplex`);
    `dtype` is defined below since the two test classes use different dtype
    sets.
    '''

    @pytest.fixture(scope='class', params=DTYPES)
    @classmethod
    def dtype(cls, request):
        return request.param

    @pytest.fixture(scope='class')
    @classmethod
    def _variant_file(cls, testdir, interleave, dtype, byteorder):
        '''Writes "92AV3C.lan" using the given interleave/dtype/byteorder
        and returns the new header file path. Written once per combination
        (class scope) and reused for both the memmap and direct-read
        variants, and across all test methods.
        '''
        image = spy.open_image('92AV3C.lan')
        fname = os.path.join(
            testdir, 'av3c_%s_%s_%s.hdr' % (interleave, dtype, byteorder))
        # `force=True`: pytest's fixture-reordering optimizer can't always
        # perfectly nest four independent class-scoped parametrize axes, so
        # this fixture is occasionally (rarely) re-invoked for a
        # (interleave, dtype, byteorder) combination that was already
        # written earlier in the run; overwriting with identical content is
        # harmless.
        spy.envi.save_image(fname, image, interleave=interleave,
                            dtype=dtype, byteorder=byteorder, force=True)
        return fname

    @pytest.fixture
    def spyfile_case(self, _variant_file, use_memmap):
        image = spy.open_image(_variant_file)
        if use_memmap:
            if not image.using_memmap:
                pytest.skip('Image does not use memmap for this '
                           'interleave/dtype/byteorder combination.')
        else:
            image._disable_memmap()
        return (image, REAL_FILE_DATUM, REAL_FILE_VALUE)


class TestSpyFileComplex(_SpyFileReadTests):
    '''Tests reading synthetically generated complex-valued data after
    saving it with every combination of interleave, dtype, and byte order,
    with and without memmap access (where the combination supports
    memmap). Only complex dtypes supported by both numpy and ENVI are
    tested.

    `interleave`, `byteorder`, and `use_memmap` are the class-scoped
    fixtures shared with `TestSpyFile`, defined above.
    '''

    @pytest.fixture(scope='class', params=COMPLEX_DTYPES)
    @classmethod
    def dtype(cls, request):
        return request.param

    @pytest.fixture(scope='class')
    @classmethod
    def _variant_file(cls, testdir, interleave, dtype, byteorder):
        '''Writes a random complex-valued array using the given
        interleave/dtype/byteorder and returns `(fname, datum, value)`.
        Written once per combination (class scope) and reused for both
        the memmap and direct-read variants, and across all test methods.
        '''
        (X, datum, value) = _complex_source(dtype)
        fname = os.path.join(
            testdir, 'complex_%s_%s_%s.hdr' % (interleave, dtype, byteorder))
        # See the comment on `TestSpyFile._variant_file` re: `force=True`.
        spy.envi.save_image(fname, X, interleave=interleave, dtype=dtype,
                            byteorder=byteorder, force=True)
        return (fname, datum, value)

    @pytest.fixture
    def spyfile_case(self, _variant_file, use_memmap):
        (fname, datum, value) = _variant_file
        image = spy.open_image(fname)
        if use_memmap:
            if not image.using_memmap:
                pytest.skip('Image does not use memmap for this '
                           'interleave/dtype/byteorder combination.')
        else:
            image._disable_memmap()
        return (image, datum, value)
