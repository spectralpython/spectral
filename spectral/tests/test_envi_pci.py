'''Reject PCI metadata instead of reading it as a standard ENVI image.'''

import numpy as np
import pytest

import spectral as spy
from spectral.io import envi


@pytest.fixture
def pci_header(tmp_path):
    header = tmp_path / 'image.pix.hdr'
    header.write_text(
        'ENVI\n'
        'samples = 3\n'
        'lines = 2\n'
        'bands = 1\n'
        'header offset = 0\n'
        'file type = PCI\n'
        'data type = 12\n'
        'interleave = BIP\n'
        'byte order = 0\n'
        'read procedures = {pci_read_spatial, pci_read_spectral}\n'
    )
    return header


@pytest.mark.parametrize('data_exists', [False, True])
@pytest.mark.parametrize('entry', ['discover', 'explicit', 'open_image'])
def test_open_pci_fails_before_reading_data(pci_header, data_exists, entry):
    image = pci_header.with_suffix('')
    # Arbitrary bytes suffice: no PCI payload should reach a raw reader.
    payload = bytes(range(16))
    if data_exists:
        image.write_bytes(payload)
    with pytest.raises(envi.EnviFeatureNotSupported, match='PCI'):
        if entry == 'open_image':
            spy.open_image(str(pci_header))
        else:
            envi.open(str(pci_header), str(image) if entry == 'explicit' else None)
    assert image.exists() == data_exists
    if data_exists:
        assert image.read_bytes() == payload


def test_read_pci_header_still_allowed(pci_header):
    header = envi.read_envi_header(str(pci_header))
    assert header['file type'] == 'PCI'
    assert header['read procedures'] == ['pci_read_spatial', 'pci_read_spectral']


@pytest.mark.parametrize('file_type', ['ENVI Standard', 'ENVI Classification', None])
def test_open_non_pci_image_unchanged(tmp_path, file_type):
    header = tmp_path / 'image.hdr'
    data = np.arange(6, dtype=np.uint16).reshape((2, 3, 1))
    envi.save_image(str(header), data)
    text = header.read_text()
    text = text.replace('file type = ENVI Standard\n',
                        '' if file_type is None else 'file type = %s\n' % file_type)
    header.write_text(text)
    image = envi.open(str(header))
    assert image.metadata.get('file type') == file_type
    np.testing.assert_array_equal(np.asarray(image.load()), data)
