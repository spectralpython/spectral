'''Tests of AVIRIS image file handling.

`aviris.open()` is exercised here against a small synthetic AVIRIS-format
file built on the fly (the format's fixed `614` columns / `224` bands means
even a single-row file is a couple hundred KB, so nothing is bundled under
`tests/data/`). AVIRIS data is always stored big-endian regardless of host
platform, so the synthetic pixel data is written explicitly as `>i2`.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_aviris.py
'''

import os

import numpy as np
from numpy.testing import assert_allclose
import pytest

from spectral.io import aviris
from spectral.io.aviris import read_aviris_bands
from spectral.io.spyfile import InvalidFileError

NCOLS = 614
NBANDS = 224
BYTES_PER_ROW = NCOLS * NBANDS * 2  # int16 pixels

AVIRIS_BAND_FILE = os.path.join(os.path.split(__file__)[0], 'data/92AV3C.spc')


def make_aviris_file(path, nrows):
    '''Writes a synthetic `nrows`-row AVIRIS-format image to `path`.

    Pixel value at (row, col, band) is `row * 1000 + band` (independent of
    `col`), which is enough to verify row/band addressing without needing
    fully unique values across all three dimensions.
    '''
    data = np.empty((nrows, NCOLS, NBANDS), dtype='>i2')
    for row in range(nrows):
        data[row, :, :] = row * 1000 + np.arange(NBANDS)
    data.tofile(path)


@pytest.fixture
def aviris_file(testdir):
    path = os.path.join(testdir, 'synthetic_aviris.img')
    make_aviris_file(path, nrows=2)
    return path


class TestAvirisOpen:

    def test_open_returns_expected_shape_and_metadata(self, aviris_file):
        img = aviris.open(aviris_file)
        assert img.shape == (2, NCOLS, NBANDS)
        assert img.scale_factor == 10000.0
        assert img.metadata['default bands'] == ['29', '18', '8']

    def test_pixel_values(self, aviris_file):
        img = aviris.open(aviris_file)
        pixel = img.read_pixel(1, 300)
        assert_allclose(pixel[0], 1000 / img.scale_factor)
        assert_allclose(pixel[223], 1223 / img.scale_factor)

        pixel0 = img.read_pixel(0, 50)
        assert_allclose(pixel0[10], 10 / img.scale_factor)

    def test_invalid_file_size_raises(self, testdir):
        path = os.path.join(testdir, 'bad_size.img')
        with open(path, 'wb') as f:
            f.write(b'\\x00' * (BYTES_PER_ROW - 1))
        with pytest.raises(InvalidFileError):
            aviris.open(path)

    def test_band_file_kwarg(self, aviris_file):
        img = aviris.open(aviris_file, band_file=AVIRIS_BAND_FILE)
        expected = read_aviris_bands(AVIRIS_BAND_FILE)
        assert list(img.bands.centers) == list(expected.centers)
        assert list(img.bands.bandwidths) == list(expected.bandwidths)

    def test_no_band_file_leaves_bands_unset(self, aviris_file):
        img = aviris.open(aviris_file)
        assert img.bands.centers is None
