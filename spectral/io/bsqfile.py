#########################################################################
#
#   bsqfile.py - This file is part of the Spectral Python (SPy) package.
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
Tools for handling files that are band sequential (BSQ).
'''

from __future__ import division, print_function, unicode_literals

import numpy as np
from .spyfile import SpyFile, MemmapFile
from spectral.utilities.python23 import typecode, tobytes, frombytes

byte_typecode = typecode('b')


class BsqFile(SpyFile, MemmapFile):
    '''
    A class to represent image files stored with bands sequential.
    '''

    def __init__(self, params, metadata=None):
        import spectral
        self.interleave = spectral.BSQ
        if metadata is None:
            metadata = {}
        SpyFile.__init__(self, params, metadata)

        self._memmap = self._open_memmap('r')

    def _open_memmap(self, mode):
        import os
        import sys
        if (os.path.getsize(self.filename) < sys.maxsize):
            try:
                (R, C, B) = self.shape
                return np.memmap(self.filename, dtype=self.dtype, mode=mode,
                                 offset=self.offset, shape=(B, R, C))
            except:
                print('Unable to create memmap interface.')
                return None
        else:
            return None


    def read_band(self, band, use_memmap=True):
        '''Reads a single band from the image.

        Arguments:

            `band` (int):

                Index of band to read.

            `use_memmap` (bool, default True):

                Specifies whether the file's memmap interface should be used
                to read the data. Setting this arg to True only has an effect
                if a memmap is being used (i.e., if `img.using_memmap` is True).
                
        Returns:

           :class:`numpy.ndarray`

                An `MxN` array of values for the specified band.
        '''
        from array import array

        if self._memmap is not None and use_memmap is True:
            data = np.array(self._memmap[band, :, :])
            if self.scale_factor != 1:
                data = data / float(self.scale_factor)
            return data

        vals = array(byte_typecode)
        offset = self.offset + band * self.sample_size * \
            self.nrows * self.ncols

        f = self.fid

        # Pixel format is BSQ, so read the whole band at once.
        f.seek(offset, 0)
        vals.fromfile(f, self.nrows * self.ncols * self.sample_size)

        arr = np.fromstring(tobytes(vals), dtype=self.dtype)
        arr = arr.reshape(self.nrows, self.ncols)

        if self.scale_factor != 1:
            return arr / float(self.scale_factor)
        return arr

    def read_bands(self, bands, use_memmap=False):
        '''Reads multiple bands from the image.

        Arguments:

            `bands` (list of ints):

                Indices of bands to read.

            `use_memmap` (bool, default False):

                Specifies whether the file's memmap interface should be used
                to read the data. Setting this arg to True only has an effect
                if a memmap is being used (i.e., if `img.using_memmap` is True).
                
        Returns:

           :class:`numpy.ndarray`

                An `MxNxL` array of values for the specified bands. `M` and `N`
                are the number of rows & columns in the image and `L` equals
                len(`bands`).
        '''

        from array import array

        if self._memmap is not None and use_memmap is True:
            data = np.array(self._memmap[bands, :, :]).transpose((1, 2, 0))
            if self.scale_factor != 1:
                data = data / float(self.scale_factor)
            return data

        f = self.fid

        arr = np.zeros((self.nrows, self.ncols, len(bands)), dtype=self.dtype)

        for j in range(len(bands)):

            vals = array(byte_typecode)
            offset = self.offset + (bands[j]) * self.sample_size \
                * self.nrows * self.ncols

            # Pixel format is BSQ, so read an entire band at time.
            f.seek(offset, 0)
            vals.fromfile(f, self.nrows * self.ncols * self.sample_size)

            band = np.fromstring(tobytes(vals), dtype=self.dtype)
            arr[:, :, j] = band.reshape(self.nrows, self.ncols)

        if self.scale_factor != 1:
            return arr / float(self.scale_factor)
        return arr

    def read_pixel(self, row, col, use_memmap=True):
        '''Reads the pixel at position (row,col) from the file.

        Arguments:

            `row`, `col` (int):

                Indices of the row & column for the pixel

            `use_memmap` (bool, default True):

                Specifies whether the file's memmap interface should be used
                to read the data. Setting this arg to True only has an effect
                if a memmap is being used (i.e., if `img.using_memmap` is True).
                
        Returns:

           :class:`numpy.ndarray`

                A length-`B` array, where `B` is the number of image bands.
        '''

        from array import array

        if self._memmap is not None and use_memmap is True:
            data = np.array(self._memmap[:, row, col])
            if self.scale_factor != 1:
                data = data / float(self.scale_factor)
            return data

        vals = array(byte_typecode)
        delta = self.sample_size * (self.nbands - 1)
        offset = self.offset + row * self.nbands * self.ncols \
            * self.sample_size + col * self.sample_size

        f = self.fid
        nPixels = self.nrows * self.ncols

        ncols = self.ncols
        sampleSize = self.sample_size
        bandSize = sampleSize * nPixels
        rowSize = sampleSize * self.ncols

        for i in range(self.nbands):
            f.seek(self.offset
                   + i * bandSize
                   + row * rowSize
                   + col * sampleSize, 0)
            vals.fromfile(f, sampleSize)

        pixel = np.fromstring(tobytes(vals), dtype=self.dtype)

        if self.scale_factor != 1:
            return pixel / float(self.scale_factor)
        return pixel

    def read_subregion(self, row_bounds, col_bounds, bands=None,
                       use_memmap=True):
        '''
        Reads a contiguous rectangular sub-region from the image.

        Arguments:

            `row_bounds` (2-tuple of ints):

                (a, b) -> Rows a through b-1 will be read.

            `col_bounds` (2-tuple of ints):

                (a, b) -> Columnss a through b-1 will be read.

            `bands` (list of ints):

                Optional list of bands to read.  If not specified, all bands
                are read.

            `use_memmap` (bool, default True):

                Specifies whether the file's memmap interface should be used
                to read the data. Setting this arg to True only has an effect
                if a memmap is being used (i.e., if `img.using_memmap` is True).
                
        Returns:

           :class:`numpy.ndarray`

                An `MxNxL` array.
        '''

        from array import array

        if self._memmap is not None and use_memmap is True:
            if bands is None:
                data = np.array(self._memmap[:, row_bounds[0]: row_bounds[1],
                                             col_bounds[0]: col_bounds[1]])
            else:
                data = np.array(
                    self._memmap[bands, row_bounds[0]: row_bounds[1],
                                 col_bounds[0]: col_bounds[1]])
            data = data.transpose((1, 2, 0))
            if self.scale_factor != 1:
                data = data / float(self.scale_factor)
            return data

        nSubRows = row_bounds[1] - row_bounds[0]  # Rows in sub-image
        nSubCols = col_bounds[1] - col_bounds[0]  # Cols in sub-image

        f = self.fid
        f.seek(self.offset, 0)

        # Increments between bands
        if bands is None:
            # Read all bands.
            bands = list(range(self.nbands))

        arr = np.zeros((nSubRows, nSubCols, len(bands)), dtype=self.dtype)

        nrows = self.nrows
        ncols = self.ncols
        sampleSize = self.sample_size
        bandSize = nrows * ncols * sampleSize
        colStartOffset = col_bounds[0] * sampleSize
        rowSize = ncols * sampleSize
        rowStartOffset = row_bounds[0] * rowSize
        nSubBands = len(bands)

        # Pixel format is BSQ
        for i in bands:
            vals = array(byte_typecode)
            bandOffset = i * bandSize
            for j in range(row_bounds[0], row_bounds[1]):
                f.seek(self.offset
                       + bandOffset
                       + j * rowSize
                       + colStartOffset, 0)
                vals.fromfile(f, nSubCols * sampleSize)
            subArray = np.fromstring(tobytes(vals),
                                     dtype=self.dtype).reshape((nSubRows,
                                                                nSubCols))
            arr[:, :, i] = subArray

        if self.scale_factor != 1:
            return arr / float(self.scale_factor)
        return arr

    def read_subimage(self, rows, cols, bands=None, use_memmap=False):
        '''
        Reads arbitrary rows, columns, and bands from the image.

        Arguments:

            `rows` (list of ints):

                Indices of rows to read.

            `cols` (list of ints):

                Indices of columns to read.

            `bands` (list of ints):

                Optional list of bands to read.  If not specified, all bands
                are read.

            `use_memmap` (bool, default False):

                Specifies whether the file's memmap interface should be used
                to read the data. Setting this arg to True only has an effect
                if a memmap is being used (i.e., if `img.using_memmap` is True).
                
        Returns:

           :class:`numpy.ndarray`

                An `MxNxL` array, where `M` = len(`rows`), `N` = len(`cols`),
                and `L` = len(bands) (or # of image bands if `bands` == None).
        '''

        from array import array

        if self._memmap is not None and use_memmap is True:
            if bands is None:
                data = np.array(self._memmap[:].take(rows, 1).take(cols, 2))
            else:
                data = np.array(
                    self._memmap.take(bands, 0).take(rows, 1).take(cols, 2))
            data = data.transpose((1, 2, 0))
            if self.scale_factor != 1:
                data = data / float(self.scale_factor)
            return data

        nSubRows = len(rows)                        # Rows in sub-image
        nSubCols = len(cols)                        # Cols in sub-image
        d_col = self.sample_size
        d_band = d_col * self.ncols
        d_row = d_band * self.nbands

        f = self.fid
        f.seek(self.offset, 0)

        # Increments between bands
        if bands is None:
            # Read all bands.
            bands = list(range(self.nbands))
        nSubBands = len(bands)

        arr = np.zeros((nSubRows, nSubCols, nSubBands), dtype=self.dtype)

        offset = self.offset
        vals = array(byte_typecode)

        nrows = self.nrows
        ncols = self.ncols
        sampleSize = self.sample_size
        bandSize = nrows * ncols * sampleSize
        sampleSize = self.sample_size
        rowSize = ncols * sampleSize

        # Pixel format is BSQ
        for i in bands:
            bandOffset = offset + i * bandSize
            for j in rows:
                rowOffset = j * rowSize
                for k in cols:
                    f.seek(bandOffset
                           + rowOffset
                           + k * sampleSize, 0)
                    vals.fromfile(f, sampleSize)
        arr = np.fromstring(tobytes(vals), dtype=self.dtype)
        arr = arr.reshape(nSubBands, nSubRows, nSubCols)
        arr = np.transpose(arr, (1, 2, 0))

        if self.scale_factor != 1:
            return arr / float(self.scale_factor)
        return arr

    def read_datum(self, i, j, k, use_memmap=True):
        '''Reads the band `k` value for pixel at row `i` and column `j`.

        Arguments:

            `i`, `j`, `k` (integer):

                Row, column and band index, respectively.

            `use_memmap` (bool, default True):

                Specifies whether the file's memmap interface should be used
                to read the data. Setting this arg to True only has an effect
                if a memmap is being used (i.e., if `img.using_memmap` is True).
                
        Using this function is not an efficient way to iterate over bands or
        pixels. For such cases, use readBands or readPixel instead.
        '''
        import array

        if self._memmap is not None and use_memmap is True:
            datum = self._memmap[k, i, j]
            if self.scale_factor != 1:
                datum /= float(self.scale_factor)
            return datum

        nrows = self.nrows
        ncols = self.ncols
        sampleSize = self.sample_size

        self.fid.seek(self.offset
                      + (k * nrows * ncols
                         + i * ncols
                         + j) * sampleSize, 0)
        vals = array.array(byte_typecode)
        vals.fromfile(self.fid, sampleSize)
        arr = np.fromstring(tobytes(vals), dtype=self.dtype)
        return arr.tolist()[0] / float(self.scale_factor)
