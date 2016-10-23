#########################################################################
#
#   aviris.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2001-2010 Thomas Boggs
#
#   Spectral Python is free software; you can redistribute it and/
#   or modify it under the terms of the GNU General Public License
#   as published by the Free Software Foundation; either version 2
#   of the License, or (at your option) any later version.
#
#   Spectral Python is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this software; if not, write to
#
#               Free Software Foundation, Inc.
#               59 Temple Place, Suite 330
#               Boston, MA 02111-1307
#               USA
#
#########################################################################
#
# Send comments to:
# Thomas Boggs, tboggs@users.sourceforge.net
#

'''
Functions for handling AVIRIS image files.
'''

from __future__ import division, print_function, unicode_literals

from warnings import warn


def open(file, band_file=None):
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
    import numpy as np
    from spectral.io.bipfile import BipFile
    import os
    import glob
    from .spyfile import find_file_path, InvalidFileError
    import spectral

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
    if spectral.byte_order != 1:
        p.dtype = np.dtype(p.dtype).newbyteorder().str
    metadata = {'default bands': ['29', '18', '8']}
    p.offset = 0

    img = BipFile(p, metadata)
    img.scale_factor = 10000.0

    if band_file:
        img.bands = read_aviris_bands(find_file_path(band_file))
    else:
        # Let user know if band cal files are available
        fileDir = os.path.split(p.filename)[0]
        calFiles = glob.glob(fileDir + '/*.spc')
        if len(calFiles) > 0:
            print('\nThe following band calibration files are located in ' \
                'the same directory as the opened AVIRIS file:\n')
            for f in calFiles:
                print("    " + os.path.split(f)[1])
            print('\nTo associate a band calibration file with an AVIRIS ' \
                  'data file, re-open the AVIRIS file with the following ' \
                  'syntax:\n')
            print('    >>> img = aviris.open(fileName, calFileName)\n')
    return img


def read_aviris_bands(cal_filename):
    '''
    Returns a BandInfo object for an AVIRIS spectral calibration file.

    Arguments:

        `cal_filename` (str):

            Name of the AVIRIS spectral calibration file.

    Returns:

        A :class:`spectral.BandInfo` object
    '''
    from spectral.utilities.python23 import IS_PYTHON3
    if IS_PYTHON3:
        import builtins
    else:
        import __builtin__ as builtins
    from spectral import BandInfo
    from .spyfile import find_file_path
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

