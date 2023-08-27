'''
Runs unit tests of image file interfaces using numpy memmaps.

The unit tests in this module assume the example file "92AV3C.lan" is in the
spectral data path.  After the file is opened it is saved in various formats
(different combinations of byte order, interleave, and data type) and for each
file written, the memmap interfaces are tested.

To run the unit tests, type the following from the system command line:

    # python -m spectral.tests.memmap
'''

from __future__ import absolute_import, division, print_function, unicode_literals

from numpy.testing import assert_almost_equal

import spectral as spy
from spectral.tests import testdir
from spectral.tests.spytest import SpyTest


class SpyFileMemmapTest(SpyTest):
    '''Tests that SpyFile memmap interfaces read and write properly.'''
    def __init__(self, file, datum, value, src_inter):
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

            `src_inter` (str):

                Interleave with which to save the source file
        '''
        self.file = file
        self.datum = tuple(datum)
        self.value = value
        self.src_inter = src_inter

    def setup(self):
        self.create_test_image_file()

    def create_test_image_file(self):
        import os
        img = spy.open_image(self.file)
        fname = os.path.join(testdir, 'memmap_test_%s.hdr' % self.src_inter)
        spy.envi.save_image(fname,
                            img,
                            dtype=img.dtype,
                            interleave=self.src_inter,
                            force=True)
        self.image = spy.open_image(fname)

    def test_spyfile_has_memmap(self):
        assert (self.image.using_memmap is True)

    def test_bip_memmap_read(self):
        self.create_test_image_file()
        (i, j, k) = self.datum
        mm = self.image.open_memmap(interleave='bip')
        assert_almost_equal(mm[i, j, k], self.value)

    def test_bil_memmap_read(self):
        self.create_test_image_file()
        (i, j, k) = self.datum
        mm = self.image.open_memmap(interleave='bil')
        assert_almost_equal(mm[i, k, j], self.value)

    def test_bsq_memmap_read(self):
        self.create_test_image_file()
        (i, j, k) = self.datum
        mm = self.image.open_memmap(interleave='bsq')
        assert_almost_equal(mm[k, i, j], self.value)

    def test_bip_memmap_write(self):
        self.create_test_image_file()
        (i, j, k) = self.datum
        mm = self.image.open_memmap(interleave='bip', writable=True)
        mm[i, j, k] = 2 * self.value
        mm.flush()
        assert_almost_equal(self.image.open_memmap()[i, j, k], 2 * self.value)

    def test_bil_memmap_write(self):
        self.create_test_image_file()
        (i, j, k) = self.datum
        mm = self.image.open_memmap(interleave='bil', writable=True)
        mm[i, k, j] = 3 * self.value
        mm.flush()
        assert_almost_equal(self.image.open_memmap()[i, j, k], 3 * self.value)

    def test_bsq_memmap_write(self):
        self.create_test_image_file()
        (i, j, k) = self.datum
        mm = self.image.open_memmap(interleave='bsq', writable=True)
        mm[k, i, j] = 3 * self.value
        mm.flush()
        assert_almost_equal(self.image.open_memmap()[i, j, k], 3 * self.value)


class SpyFileMemmapTestSuite(object):
    def __init__(self, filename, datum, value):
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
        '''
        self.filename = filename
        self.datum = datum
        self.value = value

    def run(self):
        import os
        print('\n' + '-' * 72)
        print('Running memmap tests.')
        print('-' * 72)
        if not os.path.isdir(testdir):
            os.mkdir(testdir)
        interleaves = ('bil', 'bip', 'bsq')
        for inter in interleaves:
            print('Testing memmaps with %s image file.' % inter.upper())
            test = SpyFileMemmapTest(self.filename, self.datum, self.value,
                                     inter)
            test.run()


def run():
    suite = SpyFileMemmapTestSuite('92AV3C.lan', (30, 40, 50), 5420.0)
    suite.run()


if __name__ == '__main__':
    from spectral.tests.run import parse_args, reset_stats, print_summary
    parse_args()
    reset_stats()
    run()
    print_summary()
