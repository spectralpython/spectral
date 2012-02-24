#########################################################################
#
#   bilfile.py - This file is part of the Spectral Python (SPy) package.
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
Tools for handling files that are band interleaved by line (BIL).
'''

from spyfile import SpyFile
import numpy as np

class BilFile(SpyFile):
    '''
    A class to represent image files stored with bands interleaved
    by line.
    '''

    def __init__(self, params, metadata = None):
	import sys, os
	import numpy as np
        import spectral
        self.interleave = spectral.BIL
        if metadata == None:
            metadata = {}
        SpyFile.__init__(self, params, metadata)        

	if (os.path.getsize(self.filename) < sys.maxint):
	    (R, C, B) = self.shape
	    self.memmap = np.memmap(self.filename, dtype=self.format, mode='r',
				    offset=self.offset, shape=(R,B,C))
	else:
	    self.memmap = None

    def read_band(self, band):
        '''Reads a single band from the image.
	
	Arguments:
	
	    `band` (int):
	    
		Index of band to read.
	
	Returns:
	
	   :class:`numpy.ndarray`
	   
		An `MxN` array of values for the specified band.
	'''

        from array import array
        import numpy
        
	if self.memmap != None:
	    data = np.array(self.memmap[:,band,:])
	    if self.swap:
		data.byteswap(True)
	    if self.scale_factor != 1:
		data = data / float(self.scale_factor)
	    return data

        vals = array(self.format)
        offset = self.offset + band * self.sample_size * self.ncols

        f = self.fid
        
        # Pixel format is BIL, so read an entire line at  time.
        for i in range(self.nrows):
            f.seek(offset + i * self.sample_size * self.nbands * \
                   self.ncols, 0)
            vals.fromfile(f, self.ncols)

        if self.swap:
            vals.byteswap()
        arr = numpy.array(vals.tolist())
        arr = arr.reshape((self.nrows, self.ncols))

	if self.scale_factor != 1:
	    return arr / float(self.scale_factor)
        return arr

    def read_bands(self, bands):
        '''Reads multiple bands from the image.
	
	Arguments:
	
	    `bands` (list of ints):
	    
		Indices of bands to read.
	
	Returns:
	
	   :class:`numpy.ndarray`
	   
		An `MxNxL` array of values for the specified bands. `M` and `N`
		are the number of rows & columns in the image and `L` equals
		len(`bands`).
	'''

        from array import array
        import numpy

 	if self.memmap != None:
	    data = np.array(self.memmap[:,bands,:]).transpose((0, 2, 1))
	    if self.swap:
		data.byteswap(True)
	    if self.scale_factor != 1:
		data = data / float(self.scale_factor)
	    return data

	f = self.fid

        arr = numpy.empty((self.nrows, self.ncols, len(bands)), self.format)

        for j in range(len(bands)):
  
            vals = array(self.format)
            offset = self.offset + (bands[j]) * self.sample_size * self.ncols

            # Pixel format is BIL, so read an entire line at  time.
            for i in range(self.nrows):
                f.seek(offset + i * self.sample_size * self.nbands * \
                       self.ncols, 0)
                vals.fromfile(f, self.ncols)

            if self.swap:
                vals.byteswap()
            bandArr = numpy.array(vals.tolist())
            bandArr = bandArr.reshape((self.nrows, self.ncols))
            arr[:,:,j] = bandArr

	if self.scale_factor != 1:
	    return arr / float(self.scale_factor)
        return arr


    def read_pixel(self, row, col):
        '''Reads the pixel at position (row,col) from the file.
	
	Arguments:
	
	    `row`, `col` (int):
	    
		Indices of the row & column for the pixel
	
	Returns:
	
	   :class:`numpy.ndarray`
	   
		A length-`B` array, where `B` is the number of bands in the image.
	'''

        from array import array
        import numpy
        
        vals = array(self.format)
        delta = self.sample_size * (self.nbands - 1)
        offset = self.offset + row * self.nbands * self.ncols \
                 * self.sample_size + col * self.sample_size

	if self.memmap != None:
	    data = np.array(self.memmap[row, :, col])
	    if self.swap:
		data.byteswap(True)
	    if self.scale_factor != 1:
		data = data / float(self.scale_factor)
	    return data

        f = self.fid

        ncols = self.ncols
        sampleSize = self.sample_size

        for i in range(self.nbands):
            f.seek(offset + i * sampleSize * ncols, 0)
            vals.fromfile(f, 1)

        if self.swap:
            vals.byteswap()
        pixel = numpy.array(vals.tolist(), self._typecode)

	if self.scale_factor != 1:
	    return pixel / float(self.scale_factor)
        return pixel

    def read_subregion(self, row_bounds, col_bounds, bands = None):
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
	
	Returns:
	
	   :class:`numpy.ndarray`
	   
		An `MxNxL` array.
        '''

        from array import array
        import numpy

	if self.memmap != None:
	    if bands == None:
		data = np.array(self.memmap[row_bounds[0]: row_bounds[1], :,
					    col_bounds[0]: col_bounds[1]])
	    else:
		data = np.array(self.memmap[row_bounds[0]: row_bounds[1], bands,
					    col_bounds[0]: col_bounds[1]])
	    data = data.transpose((0, 2, 1))
	    if self.swap:
		data.byteswap(True)
	    if self.scale_factor != 1:
		data = data / float(self.scale_factor)
	    return data

        nSubRows = row_bounds[1] - row_bounds[0]  # Rows in sub-image
        nSubCols = col_bounds[1] - col_bounds[0]  # Cols in sub-image
        d_row = self.sample_size * self.ncols * self.nbands
        colStartPos = col_bounds[0] * self.sample_size

        f = self.fid
        f.seek(self.offset, 0)
        
        # Increments between bands
        if bands == None:
            # Read all bands.
            bands = range(self.nbands)

        arr = numpy.empty((nSubRows, nSubCols, len(bands)), self.typecode())

        offset = self.offset
        ncols = self.ncols
        sampleSize = self.sample_size
        nSubBands = len(bands)

        # Pixel format is BIL
        for i in range(row_bounds[0], row_bounds[1]):
            f.seek(offset + i * d_row + colStartPos, 0)
            rowPos = f.tell()
            vals = array(self.format)
            for j in bands:
                f.seek(rowPos + j * ncols * sampleSize, 0)
                vals.fromfile(f, nSubCols)
            if self.swap:
                vals.byteswap()
            subArray = numpy.array(vals.tolist())
            subArray = subArray.reshape((nSubBands, nSubCols))
            arr[i - row_bounds[0],:,:] = numpy.transpose(subArray)

	if self.scale_factor != 1:
	    return arr / float(self.scale_factor)
        return arr
    

    def read_subimage(self, rows, cols, bands = None):
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
	
	Returns:
	
	   :class:`numpy.ndarray`
	   
		An `MxNxL` array, where `M` = len(`rows`), `N` = len(`cols`),
		and `L` = len(bands) (or # of image bands if `bands` == None).
        '''

        from array import array
        import numpy
        
	if self.memmap != None:
	    if bands == None:
		data = np.array(self.memmap.take(rows, 0).take(cols, 2))
	    else:
		data = np.array(self.memmap.take(rows, 0).take(bands, 1).take(cols, 2))
	    data = data.transpose((0, 2, 1))
	    if self.swap:
		data.byteswap(True)
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
        if bands == None:
            # Read all bands.
            bands = range(self.nbands)
        nSubBands = len(bands)

        arr = numpy.empty((nSubRows, nSubCols, nSubBands), self.typecode())

        offset = self.offset
        vals = array(self.format)

        # Pixel format is BIL
        for i in rows:
            for j in cols:
                for k in bands:
                    f.seek(offset +                 \
                           i * d_row +        \
                           j * d_col +        \
                           k * d_band, 0)
                    vals.fromfile(f, 1)
        if self.swap:
            vals.byteswap()
        subArray = numpy.array(vals.tolist())
        subArray = subArray.reshape((nSubRows, nSubCols, nSubBands))

	if self.scale_factor != 1:
	    return subArray / float(self.scale_factor)
        return subArray

    def read_datum(self, i, j, k):
        '''Reads the band `k` value for pixel at row `i` and column `j`.
	
	Arguments:
	
	    `i`, `j`, `k` (integer):
	    
		Row, column and band index, respectively.
	
	Using this function is not an efficient way to iterate over bands or
	pixels. For such cases, use readBands or readPixel instead.	
	'''
        import array

	if self.memmap != None:
	    datum = self.memmap[i, k, j]
	    if self.swap:
		datum = datum.byteswap()
	    if self.scale_factor != 1:
		datum /= float(self.scale_factor)
	    return datum

        d_col = self.sample_size
        d_band = d_col * self.ncols
        d_row = d_band * self.nbands

        self.fid.seek(self.offset + i * d_row + j * d_col + k * d_band, 0)
        vals = array.array(self.format)
        vals.fromfile(self.fid, 1)
        if self.swap:
            vals.byteswap()
	return vals.tolist()[0] / float(self.scale_factor)
        
