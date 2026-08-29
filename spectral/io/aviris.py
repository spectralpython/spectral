'''
Functions for handling AVIRIS image files.
'''

from __future__ import annotations

import numpy as np
import os

import builtins

import spectral as spy
from ..spectral import BandInfo
from .bipfile import BipFile
from .spyfile import find_file_path, InvalidFileError


def open(file: str, band_file: str | None = None) -> BipFile:
    '''
    Returns a SpyFile object for an AVIRIS image file.

    Arguments:

        `file` (str):

            Name of the AVIRIS data file.

        `band_file` (str):

            Optional name of the AVIRIS spectral calibration file.

    Returns:

        A SpyFile object for the image file.

    Raises:

        spectral.io.spyfile.InvalidFileError
    '''
    class Params:
        pass
    p = Params()

    p.filename = find_file_path(file)
    p.nbands = 224
    p.ncols = 614
    fileSize = os.stat(p.filename)[6]
    if fileSize % 275072 != 0:
        raise InvalidFileError('File size not consistent with AVIRIS format.')
    p.nrows = int(fileSize / 275072)
    p.byte_order = 1
    p.dtype = np.dtype('i2').str
    if spy.byte_order != 1:
        p.dtype = np.dtype(p.dtype).newbyteorder().str
    metadata = {'default bands': ['29', '18', '8']}
    p.offset = 0

    img = BipFile(p, metadata)
    img.scale_factor = 10000.0

    if band_file:
        img.bands = read_aviris_bands(find_file_path(band_file))

    return img


def read_aviris_bands(cal_filename: str) -> BandInfo:
    '''
    Returns a BandInfo object for an AVIRIS spectral calibration file.

    Arguments:

        `cal_filename` (str):

            Name of the AVIRIS spectral calibration file.

    Returns:

        A :class:`spectral.BandInfo` object
    '''
    bands = BandInfo()
    bands.band_quantity = 'Wavelength'
    bands.band_unit = 'nm'

    fin = builtins.open(find_file_path(cal_filename))
    rows = [line.split() for line in fin]
    rows = [[float(x) for x in row] for row in rows if len(row) == 5]
    columns = list(zip(*rows))
    bands.centers = columns[0]
    bands.bandwidths = columns[1]
    bands.center_stdevs = columns[2]
    bands.bandwidth_stdevs = columns[3]
    bands.band_unit = 'nm'
    return bands
