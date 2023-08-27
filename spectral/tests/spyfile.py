'''
Runs unit tests of spectral file I/O functions.

The unit tests in this module assume the example file "92AV3C.lan" is in the
spectral data path.  After the file is opened it is saved in various formats
(different combinations of byte order, interleave, and data type) and for each
file written, the new file is opened and known data values are read and checked
to verify they are read properly.

To run the unit tests, type the following from the system command line:

    # python -m spectral.tests.spyfile
'''

from __future__ import division, print_function, unicode_literals

import itertools
import numpy as np
import os
import warnings

import spectral as spy
from spectral.io.spyfile import find_file_path, FileNotFoundError, SpyFile
from spectral.tests import testdir
from spectral.tests.spytest import SpyTest

ENVI_COMPLEX_TEST_SIZES = [64, 28]

assert_almost_equal = np.testing.assert_allclose


def assert_allclose(a, b, **kwargs):
    np.testing.assert_allclose(np.array(a), np.array(b), **kwargs)


class SpyFileTest(SpyTest):
    '''Tests that SpyFile methods read data correctly from files.'''
    def __init__(self, file, datum, value):
        '''
        Arguments:

            `file` (str or `SpyFile`):

                The SpyFile to be tested.  This can be either the name of the
                file or a SpyFile object that has already been opened.

            `datum` (3-tuple of ints):

                (i, j, k) are the row, column and band of the datum to be
                tested. 'i' and 'j' should be at least 10 pixels away from the
                edge of the associated image and `k` should have at least 10
                bands above and below it in the image.

            `value` (int or float):

                The scalar value associated with location (i, j, k) in
                the image.
        '''
        self.file = file
        self.datum = datum
        self.value = value

    def setup(self):
        if isinstance(self.file, SpyFile):
            self.image = self.file
        else:
            self.image = spy.open_image(self.file)

    def test_read_datum(self):
        assert_almost_equal(self.image.read_datum(*self.datum, use_memmap=True),
                            self.value)
        assert_almost_equal(self.image.read_datum(*self.datum, use_memmap=False),
                            self.value)

    def test_read_pixel(self):
        (i, j, k) = self.datum
        assert_almost_equal(self.image.read_pixel(i, j, use_memmap=True)[k],
                            self.value)
        assert_almost_equal(self.image.read_pixel(i, j, use_memmap=False)[k],
                            self.value)

    def test_read_band(self):
        (i, j, k) = self.datum
        assert_almost_equal(self.image.read_band(k, use_memmap=True)[i, j],
                            self.value)
        assert_almost_equal(self.image.read_band(k, use_memmap=False)[i, j],
                            self.value)

    def test_read_bands(self):
        (i, j, k) = self.datum
        bands = (k - 5, k - 2, k, k + 1)
        assert_almost_equal(self.image.read_bands(bands,
                                                  use_memmap=True)[i, j, 2],
                            self.value)
        assert_almost_equal(self.image.read_bands(bands,
                                                  use_memmap=False)[i, j, 2],
                            self.value)

    def test_read_bands_nonascending(self):
        (i, j, k) = self.datum
        bands = (k - 2, k + 1, k, k - 5)
        assert_almost_equal(self.image.read_bands(bands,
                                                  use_memmap=True)[i, j, 2],
                            self.value)
        assert_almost_equal(self.image.read_bands(bands,
                                                  use_memmap=False)[i, j, 2],
                            self.value)

    def test_read_bands_duplicates(self):
        (i, j, k) = self.datum
        bands = (k - 5, k - 5, k, k - 5)
        assert_almost_equal(self.image.read_bands(bands,
                                                  use_memmap=True)[i, j, 2],
                            self.value)
        assert_almost_equal(self.image.read_bands(bands,
                                                  use_memmap=False)[i, j, 2],
                            self.value)

    def test_read_subregion(self):
        (i, j, k) = self.datum
        region = self.image.read_subregion((i - 5, i + 9),
                                           (j - 3, j + 4), use_memmap=True)
        assert_almost_equal(region[5, 3, k], self.value)
        region = self.image.read_subregion((i - 5, i + 9),
                                           (j - 3, j + 4), use_memmap=False)
        assert_almost_equal(region[5, 3, k], self.value)

    def test_read_subimage(self):
        (i, j, k) = self.datum
        subimage = self.image.read_subimage([0, 3, i, 5],
                                            [1, j, 4, 7],
                                            [3, 7, k], use_memmap=True)
        assert_almost_equal(subimage[2, 1, 2], self.value)
        subimage = self.image.read_subimage([0, 3, i, 5],
                                            [1, j, 4, 7],
                                            [3, 7, k], use_memmap=False)
        assert_almost_equal(subimage[2, 1, 2], self.value)

        subimage = self.image.read_subimage([0, 3, i, 5],
                                            [1, j, 4, 7], use_memmap=True)
        assert_almost_equal(subimage[2, 1, k], self.value)
        subimage = self.image.read_subimage([0, 3, i, 5],
                                            [1, j, 4, 7], use_memmap=False)
        assert_almost_equal(subimage[2, 1, k], self.value)

    def test_load(self):
        (i, j, k) = self.datum
        data = self.image.load()
        spyf = self.image

        load_assert = assert_allclose
        load_assert(data[i, j, k], self.value)
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

    def test_getitem_i_j_k(self):
        (i, j, k) = self.datum
        assert_almost_equal(self.image[i, j, k], self.value)

    def test_getitem_i_j(self):
        (i, j, k) = self.datum
        assert_almost_equal(self.image[i, j][k], self.value)

    def test_getitem_i_j_kslice(self):
        (i, j, k) = self.datum
        assert_almost_equal(self.image[i, j, k-2:k+3:2][0, 0, 1], self.value)

    def test_getitem_islice_jslice(self):
        (i, j, k) = self.datum
        assert_almost_equal(self.image[i-3:i+3, j-3:j+3][3, 3, k], self.value)


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

    assert_almost_equal(obj1, obj2, decimal=decimal, err_msg=err_msg,
                        verbose=verbose)


class SpyFileTestSuite(object):
    '''Tests reading by byte orders, data types, and interleaves. For a
    specified image file name, the test suite will verify proper reading of
    data for various combinations of data type, interleave (BIL, BIP, BSQ),
    and  byte order (little- and big-endian). A new file is created
    for each combination of parameters for testing.
    '''
    def __init__(self, filename, datum, value, **kwargs):
        '''
        Arguments:

            `filename` (str):

                Name of the image file to be tested.

            `datum` (3-tuple of ints):

                (i, j, k) are the row, column and band of the datum  to be
                tested. 'i' and 'j' should be at least 10 pixels away from the
                edge of the associated image and `k` should have at least 10
                bands above and below it in the image.

            `value` (int or float):

                The scalar value associated with location (i, j, k) in
                the image.

        Keyword Arguments:

            `dtypes` (tuple of numpy dtypes):

                The file will be tested for all of the dtypes given. If not
                specified, only float32 an float64 will be tested.
        '''
        self.filename = filename
        self.datum = datum
        self.value = value
        self.dtypes = kwargs.get('dtypes', ('f4', 'f8'))
        self.dtypes = [np.dtype(d).name for d in self.dtypes]

    def run(self):
        print('\n' + '-' * 72)
        print('Running SpyFile read tests.')
        print('-' * 72)

        if not os.path.isdir(testdir):
            os.mkdir(testdir)
        image = spy.open_image(self.filename)
        basename = os.path.join(testdir,
                                os.path.splitext(os.path.split(self.filename)[-1])[0])
        interleaves = ('bil', 'bip', 'bsq')
        ends = ('big', 'little')
        cases = itertools.product(interleaves, self.dtypes, ends)
        for (inter, dtype, endian) in cases:
            fname = '%s_%s_%s_%s.hdr' % (basename, inter, dtype,
                                         endian)
            spy.envi.save_image(fname, image, interleave=inter,
                                dtype=dtype, byteorder=endian)
            msg = 'Running SpyFile read tests on %s %s %s-endian file ' \
                % (inter.upper(), np.dtype(dtype).name, endian)
            testimg = spy.open_image(fname)
            if testimg.using_memmap is True:
                print('\n' + '-' * 72)
                print(msg + 'using memmap...')
                print('-' * 72)
                test = SpyFileTest(testimg, self.datum, self.value)
                test.run()
                print('\n' + '-' * 72)
                print(msg + 'without memmap...')
                print('-' * 72)
                testimg._disable_memmap()
                test = SpyFileTest(testimg, self.datum, self.value)
                test.run()
            else:
                print('\n' + '-' * 72)
                print(msg + 'without memmap...')
                print('-' * 72)
                test = SpyFileTest(testimg, self.datum, self.value)
                test.run()


def create_complex_test_files(dtypes):
    '''Create test files with complex data'''
    if not os.path.isdir(testdir):
        os.mkdir(testdir)
    tests = []
    shape = (100, 200, 64)
    datum = (33, 44, 25)
    for t in dtypes:
        X = np.array(np.random.rand(*shape) + 1j * np.random.rand(*shape),
                     dtype=t)
        fname = os.path.join(testdir, 'test_{}.hdr'.format(t))
        spy.envi.save_image(fname, X)
        tests.append((fname, datum, X[datum]))
    return tests


def run():
    tests = [('92AV3C.lan', (99, 99, 99), 2057.0)]
    for (fname, datum, value) in tests:
        try:
            find_file_path(fname)
            suite = SpyFileTestSuite(fname, datum, value,
                                     dtypes=('i2', 'i4', 'f4', 'f8', 'c8', 'c16'))
            suite.run()
        except FileNotFoundError:
            print('File "%s" not found. Skipping.' % fname)

    # Run tests for complex data types

    # Only test complex sizes supported by numpy and ENVI.
    dtypes = []
    for s in [s for s in spy.COMPLEX_SIZES if s in ENVI_COMPLEX_TEST_SIZES]:
        t = 'complex{}'.format(s)
        if hasattr(np, t):
            dtypes.append(t)
        else:
            # This is unlikely to happen because numpy currently supports
            # more complex types than ENVI
            warnings.warn('numpy does not support {}. Skipping test.'.format(t))

    tests = create_complex_test_files(dtypes)
    for (dtype, (fname, datum, value)) in zip(dtypes, tests):
        try:
            find_file_path(fname)
            suite = SpyFileTestSuite(fname, datum, value,
                                     dtypes=(dtype,))
            suite.run()
        except FileNotFoundError:
            print('File "%s" not found. Skipping.' % fname)


if __name__ == '__main__':
    run()
