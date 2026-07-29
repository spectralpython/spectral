'''
Tests of image file interfaces using numpy memmaps.

The unit tests in this module save the sample image "92AV3C.lan" in various
formats (different combinations of byte order, interleave, and data type)
and for each file written, the memmap interfaces are tested.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_memmap.py
'''

import os

from numpy.testing import assert_almost_equal
import pytest

import spectral as spy

DATUM = (30, 40, 50)
VALUE = 5420.0


@pytest.fixture
def memmap_image(testdir, src_inter, av3c_image):
    '''Saves "92AV3C.lan" using `src_inter` interleave and opens the copy.

    A fresh copy is created for every test since some tests write to the
    image via a writable memmap.
    '''
    fname = os.path.join(testdir, 'memmap_test_%s.hdr' % src_inter)
    spy.envi.save_image(fname, av3c_image, dtype=av3c_image.dtype,
                        interleave=src_inter, force=True)
    return spy.open_image(fname)


@pytest.mark.parametrize('src_inter', ['bil', 'bip', 'bsq'])
class TestSpyFileMemmap:
    '''Tests that SpyFile memmap interfaces read and write properly.'''

    def test_spyfile_has_memmap(self, memmap_image):
        assert (memmap_image.using_memmap is True)

    def test_bip_memmap_read(self, memmap_image):
        (i, j, k) = DATUM
        mm = memmap_image.open_memmap(interleave='bip')
        assert_almost_equal(mm[i, j, k], VALUE)

    def test_bil_memmap_read(self, memmap_image):
        (i, j, k) = DATUM
        mm = memmap_image.open_memmap(interleave='bil')
        assert_almost_equal(mm[i, k, j], VALUE)

    def test_bsq_memmap_read(self, memmap_image):
        (i, j, k) = DATUM
        mm = memmap_image.open_memmap(interleave='bsq')
        assert_almost_equal(mm[k, i, j], VALUE)

    def test_bip_memmap_write(self, memmap_image):
        (i, j, k) = DATUM
        mm = memmap_image.open_memmap(interleave='bip', writable=True)
        mm[i, j, k] = 2 * VALUE
        mm.flush()
        assert_almost_equal(memmap_image.open_memmap()[i, j, k], 2 * VALUE)

    def test_bil_memmap_write(self, memmap_image):
        (i, j, k) = DATUM
        mm = memmap_image.open_memmap(interleave='bil', writable=True)
        mm[i, k, j] = 3 * VALUE
        mm.flush()
        assert_almost_equal(memmap_image.open_memmap()[i, j, k], 3 * VALUE)

    def test_bsq_memmap_write(self, memmap_image):
        (i, j, k) = DATUM
        mm = memmap_image.open_memmap(interleave='bsq', writable=True)
        mm[k, i, j] = 3 * VALUE
        mm.flush()
        assert_almost_equal(memmap_image.open_memmap()[i, j, k], 3 * VALUE)
