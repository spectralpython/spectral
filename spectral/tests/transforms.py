'''
Runs unit tests for linear transforms of spectral data & data files.

The unit tests in this module assume the example file "92AV3C.lan" is in the
spectral data path.  After the file is opened, unit tests verify that
LinearTransform objects created with SpyFile and numpy.ndarray objects yield
the correct values for known image data values.

To run the unit tests, type the following from the system command line:

    # python -m spectral.tests.transforms
'''

from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy.testing import assert_almost_equal

import spectral as spy
from spectral.algorithms.transforms import LinearTransform
from spectral.io.spyfile import SpyFile
from spectral.tests.spytest import SpyTest


class LinearTransformTest(SpyTest):
    '''Tests that LinearTransform objects produce correct values.'''
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
        elif isinstance(self.file, np.ndarray):
            self.image = self.file
        else:
            self.image = spy.open_image(self.file)

        self.scalar = 10.
        self.matrix = self.scalar * np.identity(self.image.shape[2],
                                                dtype='f8')
        self.pre = 37.
        self.post = 51.

    def test_scalar_multiply(self):
        (i, j, k) = self.datum
        transform = LinearTransform(self.scalar)
        result = transform(self.image[i, j])[k]
        assert_almost_equal(result,
                            self.scalar * self.value)

    def test_pre_scalar_multiply(self):
        (i, j, k) = self.datum
        transform = LinearTransform(self.scalar, pre=self.pre)
        result = transform(self.image[i, j])[k]
        assert_almost_equal(result,
                            self.scalar * (self.pre + self.value))

    def test_scalar_multiply_post(self):
        (i, j, k) = self.datum
        transform = LinearTransform(self.scalar, post=self.post)
        result = transform(self.image[i, j])[k]
        assert_almost_equal(result,
                            self.scalar * self.value + self.post)

    def test_pre_scalar_multiply_post(self):
        (i, j, k) = self.datum
        transform = LinearTransform(self.scalar, pre=self.pre,
                                    post=self.post)
        result = transform(self.image[i, j])[k]
        assert_almost_equal(result,
                            self.scalar * (self.pre + self.value)
                            + self.post)

    def test_matrix_multiply(self):
        (i, j, k) = self.datum
        transform = LinearTransform(self.matrix)
        result = transform(self.image[i, j])[k]
        assert_almost_equal(result,
                            self.scalar * self.value)

    def test_pre_matrix_multiply(self):
        (i, j, k) = self.datum
        transform = LinearTransform(self.matrix, pre=self.pre)
        result = transform(self.image[i, j])[k]
        assert_almost_equal(result,
                            self.scalar * (self.pre + self.value))

    def test_matrix_multiply_post(self):
        (i, j, k) = self.datum
        transform = LinearTransform(self.matrix, post=self.post)
        result = transform(self.image[i, j])[k]
        assert_almost_equal(result,
                            self.scalar * self.value + self.post)

    def test_pre_matrix_multiply_post(self):
        (i, j, k) = self.datum
        transform = LinearTransform(self.matrix, pre=self.pre,
                                    post=self.post)
        result = transform(self.image[i, j])[k]
        assert_almost_equal(result,
                            self.scalar * (self.pre + self.value)
                            + self.post)


def run():
    (fname, datum, value) = ('92AV3C.lan', (99, 99, 99), 2057.0)
    image = spy.open_image(fname)
    print('\n' + '-' * 72)
    print('Running LinearTransform tests on SpyFile object.')
    print('-' * 72)
    test = LinearTransformTest(image, datum, value)
    test.run()
    data = image.load()
    print('\n' + '-' * 72)
    print('Running LinearTransform tests on ImageArray object.')
    print('-' * 72)
    test = LinearTransformTest(data, datum, value)
    test.run()
    image.scale_factor = 10000.0
    print('\n' + '-' * 72)
    print('Running LinearTransform tests on SpyFile object with scale factor.')
    print('-' * 72)
    test = LinearTransformTest(image, datum, value / 10000.0)
    test.run()

if __name__ == '__main__':
    from spectral.tests.run import parse_args, reset_stats, print_summary
    parse_args()
    reset_stats()
    run()
    print_summary()
