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
from spectral.algorithms.transforms import LinearTransform
from spectral.io.spyfile import (FileNotFoundError as SpyFileNotFoundError,
                                 SubImage, TransformedImage, find_file_path,
                                 interleave_transpose, tile_image,
                                 transform_image)

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
    def dtype(self, request):
        return request.param

    @pytest.fixture(scope='class')
    def _variant_file(self, testdir, interleave, dtype, byteorder):
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
    def dtype(self, request):
        return request.param

    @pytest.fixture(scope='class')
    def _variant_file(self, testdir, interleave, dtype, byteorder):
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


class TestInterleaveTranspose:
    '''Verifies `interleave_transpose` against ground truth built directly
    via `numpy.transpose`, independent of the function under test. `R`,
    `C`, and `B` are chosen distinct so an incorrect axis mapping would
    show up as a shape or data mismatch rather than accidentally passing.
    '''
    R, C, B = 5, 3, 7

    @pytest.fixture(scope='class')
    def forms(self):
        canonical = np.arange(self.R * self.C * self.B).reshape(
            self.R, self.C, self.B)  # bip-shaped: (R, C, B)
        return {
            'bip': canonical,
            'bil': np.transpose(canonical, (0, 2, 1)),  # (R, B, C)
            'bsq': np.transpose(canonical, (2, 0, 1)),  # (B, R, C)
        }

    @pytest.mark.parametrize('src', INTERLEAVES)
    @pytest.mark.parametrize('dst', INTERLEAVES)
    def test_transpose_matches_ground_truth(self, forms, src, dst):
        result = np.transpose(forms[src], interleave_transpose(src, dst))
        assert_allclose(result, forms[dst])

    def test_invalid_first_interleave_raises(self):
        with pytest.raises(ValueError):
            interleave_transpose('bad', 'bip')

    def test_invalid_second_interleave_raises(self):
        with pytest.raises(ValueError):
            interleave_transpose('bip', 'bad')


class TestFindFilePath:

    def test_raises_for_missing_file(self):
        with pytest.raises(SpyFileNotFoundError):
            find_file_path('this_file_does_not_exist_xyz.hdr')

    def test_finds_file_via_spectral_data_env_var(self, testdir, monkeypatch):
        fname = 'findable_test_file.txt'
        with open(os.path.join(testdir, fname), 'w') as f:
            f.write('data')
        monkeypatch.setenv('SPECTRAL_DATA', testdir)
        path = find_file_path(fname)
        assert os.path.samefile(path, os.path.join(testdir, fname))


SUB_ROW_RANGE = (90, 120)
SUB_COL_RANGE = (85, 130)


@pytest.fixture
def sub_image_case(av3c_image):
    sub = SubImage(av3c_image, list(SUB_ROW_RANGE), list(SUB_COL_RANGE))
    return (av3c_image, sub)


class TestSubImage:
    '''Tests SubImage, a SpyFile-like view onto a rectangular region of a
    parent image. Every read method is checked against the equivalent read
    on the parent image, with row/col offsets added manually.
    '''

    def test_shape(self, sub_image_case):
        (full, sub) = sub_image_case
        assert sub.shape == (SUB_ROW_RANGE[1] - SUB_ROW_RANGE[0],
                             SUB_COL_RANGE[1] - SUB_COL_RANGE[0],
                             full.nbands)

    @pytest.mark.parametrize('row_range,col_range', [
        ((-1, 10), (0, 10)),
        ((0, 100000), (0, 10)),
        ((0, 10), (-1, 10)),
        ((0, 10), (0, 100000)),
    ])
    def test_construction_out_of_range_raises(self, av3c_image, row_range,
                                              col_range):
        with pytest.raises(IndexError):
            SubImage(av3c_image, list(row_range), list(col_range))

    def test_del_after_failed_construction_does_not_raise(self, av3c_image):
        '''A SubImage that fails its row/col range validation never reaches
        `SpyFile.set_params` and so never gets a `fid` attribute; garbage
        collecting it should not raise/warn out of `SpyFile.__del__`.'''
        import gc
        try:
            SubImage(av3c_image, [-1, 10], [0, 10])
        except IndexError:
            pass
        gc.collect()

    def test_read_pixel(self, sub_image_case):
        (full, sub) = sub_image_case
        assert_allclose(
            sub.read_pixel(5, 7),
            full.read_pixel(SUB_ROW_RANGE[0] + 5, SUB_COL_RANGE[0] + 7))

    def test_read_band(self, sub_image_case):
        (full, sub) = sub_image_case
        band = 50
        expected = full.read_subregion(list(SUB_ROW_RANGE),
                                       list(SUB_COL_RANGE), [band])[:, :, 0]
        assert_allclose(sub.read_band(band), expected)

    def test_read_bands(self, sub_image_case):
        (full, sub) = sub_image_case
        bands = [10, 50, 100]
        expected = full.read_subregion(list(SUB_ROW_RANGE),
                                       list(SUB_COL_RANGE), bands)
        assert_allclose(sub.read_bands(bands), expected)

    def test_read_subregion(self, sub_image_case):
        (full, sub) = sub_image_case
        region = sub.read_subregion([2, 8], [3, 9])
        expected = full.read_subregion(
            [SUB_ROW_RANGE[0] + 2, SUB_ROW_RANGE[0] + 8],
            [SUB_COL_RANGE[0] + 3, SUB_COL_RANGE[0] + 9])
        assert_allclose(region, expected)

    def test_read_subregion_with_bands(self, sub_image_case):
        (full, sub) = sub_image_case
        bands = [4, 8]
        region = sub.read_subregion([2, 8], [3, 9], bands)
        expected = full.read_subregion(
            [SUB_ROW_RANGE[0] + 2, SUB_ROW_RANGE[0] + 8],
            [SUB_COL_RANGE[0] + 3, SUB_COL_RANGE[0] + 9], bands)
        assert_allclose(region, expected)

    def test_read_subimage(self, sub_image_case):
        (full, sub) = sub_image_case
        rows = [1, 3, 5]
        cols = [2, 4]
        result = sub.read_subimage(rows, cols)
        expected = full.read_subimage(
            [SUB_ROW_RANGE[0] + r for r in rows],
            [SUB_COL_RANGE[0] + c for c in cols])
        assert_allclose(result, expected)

    def test_read_subimage_with_bands(self, sub_image_case):
        (full, sub) = sub_image_case
        rows = [1, 3]
        cols = [2, 4]
        bands = [5, 9, 12]
        result = sub.read_subimage(rows, cols, bands)
        expected = full.read_subimage(
            [SUB_ROW_RANGE[0] + r for r in rows],
            [SUB_COL_RANGE[0] + c for c in cols], bands)
        assert_allclose(result, expected)

    def test_getitem_pixel(self, sub_image_case):
        (full, sub) = sub_image_case
        assert_allclose(sub[5, 7],
                        full[SUB_ROW_RANGE[0] + 5, SUB_COL_RANGE[0] + 7])

    def test_getitem_datum(self, sub_image_case):
        (full, sub) = sub_image_case
        assert_allclose(
            sub[5, 7, 50],
            full[SUB_ROW_RANGE[0] + 5, SUB_COL_RANGE[0] + 7, 50])

    def test_getitem_region_slice(self, sub_image_case):
        (full, sub) = sub_image_case
        result = sub[2:8, 3:9]
        expected = full[SUB_ROW_RANGE[0] + 2:SUB_ROW_RANGE[0] + 8,
                       SUB_COL_RANGE[0] + 3:SUB_COL_RANGE[0] + 9]
        assert_allclose(result, expected)

    def test_load(self, sub_image_case):
        (full, sub) = sub_image_case
        loaded = sub.load()
        expected = full.read_subregion(list(SUB_ROW_RANGE),
                                       list(SUB_COL_RANGE))
        assert_allclose(loaded, expected)

    def test_load_dtype_kwarg(self, sub_image_case):
        (full, sub) = sub_image_case
        loaded = sub.load(dtype='f8')
        assert loaded.dtype == np.dtype('f8')

    def test_load_scale_false_raises(self, sub_image_case):
        (full, sub) = sub_image_case
        with pytest.raises(NotImplementedError):
            sub.load(scale=False)

    def test_load_invalid_kwarg_raises(self, sub_image_case):
        (full, sub) = sub_image_case
        with pytest.raises(ValueError):
            sub.load(bogus=True)


class TestTileImage:

    def test_tiles_reconstruct_original_image(self, av3c_image):
        tiles = tile_image(av3c_image, 3, 4)
        assert len(tiles) == 3
        assert all(len(row) == 4 for row in tiles)
        reconstructed = np.zeros(av3c_image.shape)
        for row_of_tiles in tiles:
            for tile in row_of_tiles:
                r0 = tile.row_offset
                r1 = r0 + tile.nrows
                c0 = tile.col_offset
                c1 = c0 + tile.ncols
                reconstructed[r0:r1, c0:c1, :] = tile.load()
        assert_allclose(reconstructed, av3c_image.load())


class TestTransformedImage:
    '''Tests TransformedImage, a lazily-transformed view of a SpyFile that
    applies a LinearTransform to each pixel as data is read.
    '''

    @pytest.fixture
    def xform(self, av3c_image):
        matrix = np.random.RandomState(0).rand(3, av3c_image.nbands)
        return LinearTransform(matrix)

    @pytest.fixture
    def timg(self, av3c_image, xform):
        return TransformedImage(xform, av3c_image)

    @pytest.fixture
    def small_timg(self, av3c_image, xform):
        '''A TransformedImage wrapping just a 5x5 corner of the image, for
        tests that would otherwise iterate pixel-by-pixel over the full
        145x145 image.'''
        sub = SubImage(av3c_image, [0, 5], [0, 5])
        return TransformedImage(xform, sub)

    def test_rejects_non_image_argument(self, xform):
        with pytest.raises(Exception):
            TransformedImage(xform, np.zeros((5, 5, 220)))

    def test_dim_mismatch_raises(self, av3c_image):
        bad_transform = LinearTransform(
            np.zeros((3, av3c_image.nbands + 1)))
        with pytest.raises(Exception):
            TransformedImage(bad_transform, av3c_image)

    def test_shape(self, timg, av3c_image):
        assert timg.shape == (av3c_image.nrows, av3c_image.ncols, 3)

    def test_bands_property_delegates_to_wrapped_image(self, timg,
                                                        av3c_image):
        assert timg.bands is av3c_image.bands

    def test_read_pixel(self, timg, xform, av3c_image):
        (i, j, k) = REAL_FILE_DATUM
        expected = xform(av3c_image.read_pixel(i, j))
        assert_allclose(timg.read_pixel(i, j), expected)

    def test_read_datum(self, timg, xform, av3c_image):
        (i, j, k) = REAL_FILE_DATUM
        expected = xform(av3c_image.read_pixel(i, j))[0]
        assert_allclose(timg.read_datum(i, j, 0), expected)

    def test_load(self, timg, xform, av3c_image):
        expected = xform(av3c_image.load())
        assert_allclose(timg.load(), expected)

    def test_read_subregion(self, timg, xform, av3c_image):
        region = timg.read_subregion((10, 20), (30, 40))
        expected = xform(av3c_image.read_subregion((10, 20), (30, 40)))
        assert_allclose(region, expected)

    def test_read_subregion_with_bands(self, timg, xform, av3c_image):
        region = timg.read_subregion((10, 20), (30, 40), [0, 2])
        expected = xform(av3c_image.read_subregion((10, 20), (30, 40)))
        assert_allclose(region, expected[:, :, [0, 2]])

    def test_read_subimage(self, timg, xform, av3c_image):
        (rows, cols) = ([10, 15, 20], [30, 35])
        result = timg.read_subimage(rows, cols)
        expected = xform(av3c_image.read_subimage(rows, cols))
        assert_allclose(result, expected)

    def test_read_subimage_with_bands(self, timg, xform, av3c_image):
        (rows, cols) = ([10, 15], [30, 35])
        result = timg.read_subimage(rows, cols, [0, 2])
        expected = xform(av3c_image.read_subimage(rows, cols))
        assert_allclose(result, expected[:, :, [0, 2]])

    def test_read_bands(self, small_timg, xform):
        bands = [0, 2]
        result = small_timg.read_bands(bands)
        expected = np.zeros((5, 5, len(bands)))
        for i in range(5):
            for j in range(5):
                expected[i, j] = small_timg.read_pixel(i, j)[bands]
        assert_allclose(result, expected)

    def test_str(self, timg):
        s = str(timg)
        assert 'TransformedImage' in s
        assert '# Bands' in s

    def test_double_wrap_chains_transforms(self, av3c_image):
        '''Wrapping a TransformedImage in another TransformedImage should
        collapse to directly wrapping the original image with a single
        chained transform (see TransformedImage.__init__), rather than
        nesting. This also exercises LinearTransform.chain() end-to-end.
        '''
        t1 = LinearTransform(np.random.RandomState(1).rand(5, av3c_image.nbands))
        t2 = LinearTransform(np.random.RandomState(2).rand(3, 5))
        inner = TransformedImage(t1, av3c_image)
        outer = TransformedImage(t2, inner)

        assert outer.image is av3c_image

        (i, j, k) = REAL_FILE_DATUM
        expected = t2(t1(av3c_image.read_pixel(i, j)))
        assert_allclose(outer.read_pixel(i, j), expected)


class TestTransformImageFunction:
    '''Tests the module-level `transform_image` function, which
    `SpyFile.transform` delegates to.'''

    def test_ndarray_with_linear_transform(self, av3c_image):
        data = av3c_image.read_subregion([0, 5], [0, 5])
        xform = LinearTransform(np.eye(av3c_image.nbands) * 2)
        result = transform_image(xform, data)
        assert_allclose(result, data * 2)

    def test_ndarray_with_plain_matrix(self, av3c_image):
        data = av3c_image.read_subregion([0, 3], [0, 3])
        matrix = np.eye(av3c_image.nbands) * 3
        result = transform_image(matrix, data)
        assert_allclose(result, data * 3)

    def test_spyfile_returns_transformed_image(self, av3c_image):
        xform = LinearTransform(np.eye(av3c_image.nbands))
        result = transform_image(xform, av3c_image)
        assert isinstance(result, TransformedImage)

    def test_spyfile_transform_method(self, av3c_image):
        '''SpyFile.transform() is a thin wrapper around transform_image().'''
        xform = LinearTransform(np.eye(av3c_image.nbands) * 2)
        result = av3c_image.transform(xform)
        assert isinstance(result, TransformedImage)
        (i, j, k) = REAL_FILE_DATUM
        assert_allclose(result.read_pixel(i, j),
                        2 * av3c_image.read_pixel(i, j))
