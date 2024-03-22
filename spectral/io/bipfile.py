'''
Code for handling files that are band interleaved by pixel (BIP).
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import array
import logging
import numpy as np
import os
import sys

import spectral as spy
from .spyfile import SpyFile, MemmapFile
from spectral.utilities.python23 import typecode, tobytes

byte_typecode = typecode('b')


class BipFile(SpyFile, MemmapFile):
    '''
    A class to interface image files stored with bands interleaved by pixel.
    '''
    def __init__(self, params, metadata=None):
        self.interleave = spy.BIP
        if metadata is None:
            metadata = {}
        SpyFile.__init__(self, params, metadata)

        self._memmap = self._open_memmap('r')

    def _open_memmap(self, mode):
        logger = logging.getLogger('spectral')
        if (os.path.getsize(self.filename) < sys.maxsize):
            try:
                (R, C, B) = self.shape
                return np.memmap(self.filename, dtype=self.dtype, mode=mode,
                                 offset=self.offset, shape=self.shape)
            except:
                logger.debug('Unable to create memmap interface.')
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
        if self._memmap is not None and use_memmap is True:
            data = np.array(self._memmap[:, :, band])
            if self.scale_factor != 1:
                data = data / float(self.scale_factor)
            return data

        vals = array.array(byte_typecode)
        delta = self.sample_size * (self.nbands - 1)
        nVals = self.nrows * self.ncols
        sample_size = self.sample_size

        f = self.fid

        f.seek(self.offset + self.sample_size * band, 0)

        # Pixel format is BIP
        for i in range(nVals - 1):
            vals.fromfile(f, sample_size)
            f.seek(delta, 1)
        vals.fromfile(f, sample_size)

        arr = np.frombuffer(tobytes(vals), dtype=self.dtype)
        arr = arr.reshape(self.nrows, self.ncols)

        if self.scale_factor != 1:
            return arr / float(self.scale_factor)
        return arr

    def read_bands(self, bands, use_memmap=True):
        '''Reads multiple bands from the image.

        Arguments:

            `bands` (list of ints):

                Indices of bands to read.

            `use_memmap` (bool, default True):

                Specifies whether the file's memmap interface should be used
                to read the data. Setting this arg to True only has an effect
                if a memmap is being used (i.e., if `img.using_memmap` is True).

        Returns:

           :class:`numpy.ndarray`

                An `MxNxL` array of values for the specified bands. `M` and `N`
                are the number of rows & columns in the image and `L` equals
                len(`bands`).
        '''
        if self._memmap is not None and use_memmap is True:
            data = np.array(self._memmap[:, :, bands])
            if self.scale_factor != 1:
                data = data / float(self.scale_factor)
            return data

        vals = array.array(byte_typecode)
        offset = self.offset
        delta = self.sample_size * self.nbands
        nVals = self.nrows * self.ncols
        sample_size = self.sample_size

        # Increments between bands
        delta_b = list(bands[:])
        for i in range(len(delta_b)):
            delta_b[i] *= self.sample_size

        f = self.fid

        # Pixel format is BIP
        for i in range(nVals):
            pixelOffset = offset + i * delta
            for j in range(len(bands)):
                f.seek(pixelOffset + delta_b[j], 0)        # Next band
                vals.fromfile(f, sample_size)
        arr = np.frombuffer(tobytes(vals), dtype=self.dtype)
        arr = arr.reshape(self.nrows, self.ncols, len(bands))

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
        if self._memmap is not None and use_memmap is True:
            data = np.array(self._memmap[row, col, :])
            if self.scale_factor != 1:
                data = data / float(self.scale_factor)
            return data

        vals = array.array(byte_typecode)

        f = self.fid
        f.seek(self.offset + self.sample_size
               * self.nbands * (row * self.ncols + col), 0)
        # Pixel format is BIP so read entire pixel.
        vals.fromfile(f, self.nbands * self.sample_size)

        pixel = np.frombuffer(tobytes(vals), dtype=self.dtype)

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

                (a, b) -> Columns a through b-1 will be read.

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
        if self._memmap is not None and use_memmap is True:
            if bands is None:
                data = np.array(self._memmap[row_bounds[0]: row_bounds[1],
                                             col_bounds[0]: col_bounds[1], :])
            else:
                data = np.array(self._memmap[row_bounds[0]: row_bounds[1],
                                             col_bounds[0]: col_bounds[1],
                                             bands])
            if self.scale_factor != 1:
                data = data / float(self.scale_factor)
            return data

        offset = self.offset
        nbands = self.nbands
        nSubRows = row_bounds[1] - row_bounds[0]  # Rows in sub-image
        nSubCols = col_bounds[1] - col_bounds[0]  # Cols in sub-image
        d_row = self.sample_size * self.ncols * self.nbands
        colStartPos = col_bounds[0] * self.sample_size * self.nbands
        vals = array.array(byte_typecode)
        sample_size = self.sample_size

        # Increments between bands
        if bands is not None:
            allBands = 0
            nSubBands = len(bands)
            delta_b = bands[:]
            for i in range(len(delta_b)):
                delta_b[i] *= self.sample_size
        else:
            allBands = 1
            nSubBands = self.nbands

        f = self.fid

        # Pixel format is BIP
        for i in range(row_bounds[0], row_bounds[1]):
            f.seek(offset + i * d_row + colStartPos, 0)
            rowPos = f.tell()

            if allBands:
                # This is the simple one
                vals.fromfile(f, nSubCols * nbands * sample_size)
            else:
                # Need to pull out specific bands for each column.
                for j in range(nSubCols):
                    f.seek(rowPos + j * self.sample_size * self.nbands, 0)
                    pixelPos = f.tell()
                    for k in range(len(bands)):
                        f.seek(pixelPos + delta_b[k], 0)    # Next band
                        vals.fromfile(f, sample_size)
        arr = np.frombuffer(tobytes(vals), dtype=self.dtype)
        arr = arr.reshape(nSubRows, nSubCols, nSubBands)

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
        if self._memmap is not None and use_memmap is True:
            if bands is None:
                data = np.array(self._memmap.take(rows, 0).take(cols, 1))
            else:
                data = np.array(
                    self._memmap.take(rows, 0).take(cols, 1).take(bands, 2))
            if self.scale_factor != 1:
                data = data / float(self.scale_factor)
            return data

        offset = self.offset
        nSubRows = len(rows)                        # Rows in sub-image
        nSubCols = len(cols)                        # Cols in sub-image
        d_band = self.sample_size
        d_col = d_band * self.nbands
        d_row = d_col * self.ncols
        vals = array.array(byte_typecode)
        sample_size = self.sample_size

        # Increments between bands
        if bands is not None:
            allBands = 0
            nSubBands = len(bands)
        else:
            allBands = 1
            bands = list(range(self.nbands))
            nSubBands = self.nbands

        f = self.fid

        # Pixel format is BIP
        for i in rows:
            for j in cols:
                if allBands:
                    f.seek(offset + i * d_row + j * d_col, 0)
                    vals.fromfile(f, nSubBands * sample_size)
                else:
                    for k in bands:
                        f.seek(offset +
                               i * d_row +
                               j * d_col +
                               k * d_band, 0)
                        vals.fromfile(f, sample_size)

        arr = np.frombuffer(tobytes(vals), dtype=self.dtype)
        arr = arr.reshape(nSubRows, nSubCols, nSubBands)

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
        if self._memmap is not None and use_memmap is True:
            datum = self._memmap[i, j, k]
            if self.scale_factor != 1:
                datum /= float(self.scale_factor)
            return datum

        vals = array.array(byte_typecode)
        f = self.fid
        f.seek(self.offset + self.sample_size
               * (self.nbands * (i * self.ncols + j) + k), 0)
        # Pixel format is BIP so read entire pixel.
        vals.fromfile(f, self.sample_size)
        arr = np.frombuffer(tobytes(vals), dtype=self.dtype)
        return arr.tolist()[0] / float(self.scale_factor)
