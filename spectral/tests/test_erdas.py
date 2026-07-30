'''Tests of ERDAS/Lan image file handling.

`erdas.open()` and `read_erdas_lan_header()` are exercised here against
small synthetic ERDAS/Lan files built on the fly, covering header/packing
variations that the single bundled sample file (`92AV3C.lan`, used
elsewhere via the `av3c_image` fixture) never exercises: the "HEADER"
(float-valued coordinate fields) header variant, unsupported/unrecognized
packing values, and the byte-order guess-then-retry fallback.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_erdas.py
'''

import os
import struct

import numpy as np
from numpy.testing import assert_allclose
import pytest

from spectral.io import erdas
from spectral.io.spyfile import InvalidFileError

# Layout of the 128-byte ERDAS/Lan header, as parsed by
# `read_erdas_lan_header`: 6-byte type field, packing/nbands (int16 each),
# 6 bytes unused, 4 coordinate fields (int32 for "HEAD74", float32 for
# "HEADER"), 56 bytes unused, map_type/nclasses (int16 each), 14 bytes
# unused, area_unit (int16), then 5 float32 fields.
_HEADER_FMT_TAIL = '6x{coords}56x2h14xh5f'


def build_erdas_header(byte_order='<', header_type=b'HEAD74', packing=2,
                       nbands=3, ncols=4, nrows=2):
    coords_fmt = '4i' if header_type == b'HEAD74' else '4f'
    coords = (ncols, nrows, 0, 0) if header_type == b'HEAD74' \
        else (float(ncols), float(nrows), 0.0, 0.0)
    fmt = byte_order + '6s2h' + _HEADER_FMT_TAIL.format(coords=coords_fmt)
    header = struct.pack(fmt, header_type, packing, nbands, *coords,
                         0, 0, 0, 0.0, 0.0, 0.0, 1.0, 1.0)
    assert len(header) == 128
    return header


def build_erdas_file(path, byte_order='<', header_type=b'HEAD74', packing=2,
                     nbands=3, ncols=4, nrows=2):
    '''Writes a synthetic ERDAS/Lan file with a BIL-ordered pixel value of
    `row * 100 + band * 10 + col` at each (row, band, col).'''
    header = build_erdas_header(byte_order, header_type, packing, nbands,
                                ncols, nrows)
    # For the unsupported/unrecognized-packing tests, `erdas.open()` raises
    # before any pixel data would be read, so the dtype used to write the
    # (never-parsed) placeholder data doesn't matter.
    dtype = {0: '<i1', 2: '<i2'}.get(packing, '<i1')
    data = np.empty((nrows, nbands, ncols), dtype=dtype)
    for row in range(nrows):
        for band in range(nbands):
            data[row, band, :] = row * 100 + band * 10 + np.arange(ncols)
    with open(path, 'wb') as f:
        f.write(header)
        data.tofile(f)


class TestErdasOpen:

    def test_open_returns_expected_shape_and_pixels(self, testdir):
        path = os.path.join(testdir, 'synthetic.lan')
        build_erdas_file(path, nbands=3, ncols=4, nrows=2)
        img = erdas.open(path)
        assert img.shape == (2, 4, 3)
        pixel = img.read_pixel(1, 2)
        assert_allclose(pixel, [102, 112, 122])

    def test_header_type_variant_with_float_coords(self, testdir):
        '''The "HEADER" (as opposed to "HEAD74") header variant stores
        image dimensions as floats rather than 32-bit integers.'''
        path = os.path.join(testdir, 'synthetic_header.lan')
        build_erdas_file(path, header_type=b'HEADER', nbands=2, ncols=3,
                         nrows=2)
        img = erdas.open(path)
        assert img.shape == (2, 3, 2)
        pixel = img.read_pixel(1, 1)
        assert_allclose(pixel, [101, 111])

    def test_8bit_packing(self, testdir):
        path = os.path.join(testdir, 'synthetic_8bit.lan')
        build_erdas_file(path, packing=0, nbands=2, ncols=3, nrows=1)
        img = erdas.open(path)
        assert np.dtype(img.dtype).itemsize == 1
        assert img.shape == (1, 3, 2)

    def test_4bit_packing_raises(self, testdir):
        path = os.path.join(testdir, 'synthetic_4bit.lan')
        build_erdas_file(path, packing=1)
        with pytest.raises(InvalidFileError):
            erdas.open(path)

    def test_unrecognized_packing_raises(self, testdir):
        path = os.path.join(testdir, 'synthetic_badpacking.lan')
        build_erdas_file(path, packing=99)
        with pytest.raises(InvalidFileError):
            erdas.open(path)

    def test_unrecognized_header_type_raises(self, testdir):
        path = os.path.join(testdir, 'synthetic_badtype.lan')
        with open(path, 'wb') as f:
            f.write(b'BADHDR' + b'\x00' * 122)
        with pytest.raises(InvalidFileError):
            erdas.open(path)

    def test_byte_order_guess_retry(self, testdir):
        '''A header written in the opposite byte order from what
        `erdas.open()` initially guesses should look invalid (huge/negative
        `nbands`/`ncols`/`nrows`), triggering a retry that re-parses the
        header with the other byte order.'''
        path = os.path.join(testdir, 'synthetic_be_header.lan')
        build_erdas_file(path, byte_order='>', nbands=3, ncols=4, nrows=2)

        # Confirm the premise: naively parsed with the default guess, the
        # header's numeric fields are indeed out of the sane range that
        # triggers `erdas.open()`'s retry.
        lh = erdas.read_erdas_lan_header(path, 0)
        assert lh['nbands'] > 512 or lh['ncols'] > 10000 or lh['nrows'] > 10000

        img = erdas.open(path)
        assert img.shape == (2, 4, 3)
        pixel = img.read_pixel(1, 2)
        assert_allclose(pixel, [102, 112, 122])
