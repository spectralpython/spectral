'''ENVI spectral library byte-layout tests, independent of the SPy writer.'''

import struct

import numpy as np
import pytest

from spectral.io import envi


# Exact Python values, including integers not representable as doubles.
LIBRARY_TYPES = [
    (1, 'B', 'u1', [0, 1, 127, 128, 254, 255]),
    (2, 'h', 'i2', [-32768, -123, 0, 1, 256, 32767]),
    (3, 'i', 'i4', [-2147483648, -123, 0, 1, 65536, 2147483647]),
    (4, 'f', 'f4', [-12.5, 0.25, 1.5, 3.75, 16.0, 1024.5]),
    (5, 'd', 'f8', [-12.5, 0.25, 1.5, 3.75, 16.0, 1024.5]),
    (6, 'ff', 'c8', [-2+3j, 0.25-0.5j, 1+2j, 3-4j, 16+0j, 0-1j]),
    (9, 'dd', 'c16', [-2+3j, 0.25-0.5j, 1+2j, 3-4j, 16+0j, 0-1j]),
    (12, 'H', 'u2', [0, 1, 255, 256, 32768, 65535]),
    (13, 'I', 'u4', [0, 1, 65535, 65536, 2147483648, 4294967295]),
    (14, 'q', 'i8', [-2**63, -2**53-1, -1, 0, 2**53+1, 2**63-1]),
    (15, 'Q', 'u8', [0, 1, 2**53+1, 2**63-1, 2**63, 2**64-1]),
]


def write_library(testdir, type_info, byte_order, offset, explicit=False):
    '''Pack raw samples without using SPy or NumPy serialization.'''
    from pathlib import Path

    code, fmt, dtype, values = type_info
    endian = '<' if byte_order == 0 else '>'
    payload = b''.join(
        struct.pack(endian + fmt, value.real, value.imag)
        if dtype.startswith('c') else struct.pack(endian + fmt, value)
        for value in values
    )
    header = Path(testdir) / 'offset.hdr'
    image = Path(testdir) / ('payload.custom' if explicit else 'offset.sli')
    prefix = b'\xa5' * (offset or 0)
    image.write_bytes(prefix + payload + b'trailing bytes are not samples')
    header.write_text(
        'ENVI\n'
        'samples = 3\n'
        'lines = 2\n'
        'bands = 1\n'
        'file type = ENVI Spectral Library\n'
        'interleave = bsq\n'
        'data type = %d\n'
        'byte order = %d\n'
        'spectra names = { first, second }\n'
        'wavelength = { 2.0, 1.0, 3.0 }\n'
        'fwhm = { 0.25, 0.5, 0.75 }\n'
        'wavelength units = Micrometers\n' % (code, byte_order)
        + ('' if offset is None else 'header offset = %d\n' % offset)
    )
    return header, image, payload


@pytest.mark.parametrize('type_info', LIBRARY_TYPES,
                         ids=[str(t[0]) for t in LIBRARY_TYPES])
@pytest.mark.parametrize('byte_order', [0, 1], ids=['little', 'big'])
@pytest.mark.parametrize('offset', [None, 0, 1, 3, 16, 1024])
@pytest.mark.parametrize('explicit', [False, True],
                         ids=['discover', 'explicit'])
def test_library_header_offset(testdir, type_info, byte_order, offset,
                               explicit):
    header, image, payload = write_library(
        testdir, type_info, byte_order, offset, explicit)
    original_bytes = image.read_bytes()
    library = envi.open(str(header), str(image) if explicit else None)
    assert isinstance(library, envi.SpectralLibrary)
    assert library.spectra.shape == (2, 3)
    expected_dtype = np.dtype(('<' if byte_order == 0 else '>') + type_info[2])
    assert library.spectra.dtype == expected_dtype
    # tolist keeps exact signed/unsigned 64-bit integers, unlike float oracles.
    assert library.spectra.tolist() == [type_info[3][:3], type_info[3][3:]]
    assert library.names == ['first', 'second']
    assert library.bands.centers == [2.0, 1.0, 3.0]
    assert library.bands.bandwidths == [0.25, 0.5, 0.75]
    assert library.bands.band_unit == 'Micrometers'
    assert library.params.offset == (offset or 0)
    assert image.read_bytes() == original_bytes


@pytest.mark.parametrize('byte_order', [0, 1])
@pytest.mark.parametrize('offset', [0, 1, 16, 1024])
def test_library_truncated_payload(testdir, byte_order, offset):
    header, image, payload = write_library(
        testdir, LIBRARY_TYPES[3], byte_order, offset)
    # A long prefix must not be mistaken for the missing final sample.
    image.write_bytes(b'\xa5' * offset + payload[:-4])
    with pytest.raises(ValueError, match='reshape'):
        envi.open(str(header))


@pytest.mark.parametrize('byte_order', [0, 1])
@pytest.mark.parametrize('offset', [None, 0, 1, 3, 16, 1024])
def test_library_save_resets_offset(testdir, byte_order, offset):
    from pathlib import Path

    header, image, payload = write_library(
        testdir, LIBRARY_TYPES[3], byte_order, offset)
    library = envi.open(str(header))
    basename = str(Path(testdir) / 'saved')
    library.save(basename)
    saved_header = envi.read_envi_header(basename + '.hdr')
    assert int(saved_header['header offset']) == 0
    raw = Path(basename + '.sli').read_bytes()
    assert len(raw) == 6 * 4
    import sys
    endian = '<' if sys.byteorder == 'little' else '>'
    assert struct.unpack(endian + '6f', raw) == tuple(LIBRARY_TYPES[3][3])
    reread = envi.open(basename + '.hdr')
    assert reread.spectra.tolist() == [
        LIBRARY_TYPES[3][3][:3], LIBRARY_TYPES[3][3][3:]]
    assert reread.names == ['first', 'second']
    assert reread.bands.centers == [2.0, 1.0, 3.0]
