#########################################################################
#
#   bipfile.py - This file is part of the Spectral Python (SPy) package.
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
Tools for handling files that are band interleaved by pixel (BIP).
'''

from spyfile import SpyFile
import numpy as np

class BipFile(SpyFile):
    '''
    A class to interface image files stored with bands interleaved by pixel.
    '''
    def __init__(self, params, metadata = None):
	import sys, os
	import numpy as np
        import spectral
        self.interleave = spectral.BIP
        if metadata == None:
            metadata = {}
        SpyFile.__init__(self, params, metadata)        

	if (os.path.getsize(self.filename) < sys.maxint):
	    self.memmap = np.memmap(self.filename, dtype=self.format, mode='r',
				    offset=self.offset, shape=self.shape)
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
        import numpy.oldnumeric as Numeric

	if self.memmap != None:
	    data = np.array(self.memmap[:,:,band])
	    if self.swap:
		data.byteswap(True)
	    if self.scale_factor != 1:
		data = data / float(self.scale_factor)
	    return data

        vals = array(self.format)
        delta = self.sample_size * (self.nbands - 1) 
        nVals = self.nrows * self.ncols

        f = self.fid

        f.seek(self.offset + self.sample_size * band, 0)
        
        # Pixel format is BIP
        for i in range(nVals - 1):
            vals.fromfile(f, 1)
            f.seek(delta, 1)
        vals.fromfile(f, 1)

        if self.swap:
            vals.byteswap()            
        arr = Numeric.array(vals.tolist())
        arr = Numeric.reshape(arr, (self.nrows, self.ncols))

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
        import numpy.oldnumeric as Numeric

	if self.memmap != None:
	    data = np.array(self.memmap[:,:,bands])
	    if self.swap:
		data.byteswap(True)
	    if self.scale_factor != 1:
		data = data / float(self.scale_factor)
	    return data

        vals = array(self.format)
        offset = self.offset
        delta = self.sample_size * self.nbands
        nVals = self.nrows * self.ncols

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
                vals.fromfile(f, 1)

        if self.swap:
            vals.byteswap()
        arr = Numeric.array(vals.tolist())
        arr = Numeric.reshape(arr, (self.nrows, self.ncols, len(bands)))

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
        import numpy.oldnumeric as Numeric

	if self.memmap != None:
	    data = np.array(self.memmap[row, col,:])
	    if self.swap:
		data.byteswap(True)
	    if self.scale_factor != 1:
		data = data / float(self.scale_factor)
	    return data

        vals = array(self.format)

        f = self.fid        
        f.seek(self.offset + self.sample_size \
               * self.nbands * (row * self.ncols + col), 0)
        # Pixel format is BIP so read entire pixel.
        vals.fromfile(f, self.nbands)

        if self.swap:
            vals.byteswap()
        pixel = Numeric.array(vals.tolist())

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
        import array
        import numpy.oldnumeric as Numeric

	if self.memmap != None:
	    if bands == None:
		data = np.array(self.memmap[row_bounds[0]: row_bounds[1],
					    col_bounds[0]: col_bounds[1], :])
	    else:
		data = np.array(self.memmap[row_bounds[0]: row_bounds[1],
					    col_bounds[0]: col_bounds[1], bands])
	    if self.swap:
		data.byteswap(True)
	    if self.scale_factor != 1:
		data = data / float(self.scale_factor)
	    return data

        offset = self.offset
        nbands = self.nbands
        nSubRows = row_bounds[1] - row_bounds[0]  # Rows in sub-image
        nSubCols = col_bounds[1] - col_bounds[0]  # Cols in sub-image
        d_row = self.sample_size * self.ncols * self.nbands
        colStartPos = col_bounds[0] * self.sample_size * self.nbands
        vals = array.array(self.format)
        nVals = self.nrows * self.ncols

        # Increments between bands
        if bands != None:
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
                vals.fromfile(f, nSubCols * nbands)
            else:
                # Need to pull out specific bands for each column.
                for j in range(nSubCols):
                    f.seek(rowPos + j * self.sample_size * self.nbands, 0)
                    pixelPos = f.tell()
                    for k in range(len(bands)):
                        f.seek(pixelPos + delta_b[k], 0)    # Next band
                        vals.fromfile(f, 1)

        if self.swap:
            vals.byteswap()
        arr = Numeric.array(vals.tolist())
        arr = Numeric.reshape(arr, (nSubRows, nSubCols, nSubBands))

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
        import array
        import numpy.oldnumeric as Numeric

	if self.memmap != None:
	    if bands == None:
		data = np.array(self.memmap.take(rows, 0).take(cols, 1))
	    else:
		data = np.array(self.memmap.take(rows, 0).take(cols, 1).take(bands, 2))
	    if self.swap:
		data.byteswap(True)
	    if self.scale_factor != 1:
		data = data / float(self.scale_factor)
	    return data

        offset = self.offset
        nbands = self.nbands
        nSubRows = len(rows)                        # Rows in sub-image
        nSubCols = len(cols)                        # Cols in sub-image
        d_band = self.sample_size
        d_col = d_band * self.nbands
        d_row = d_col * self.ncols
        vals = array.array(self.format)
        nVals = self.nrows * self.ncols

        # Increments between bands
        if bands != None:
            allBands = 0
            nSubBands = len(bands)
        else:
            allBands = 1
            bands = range(self.nbands)
            nSubBands = self.nbands

        f = self.fid
        
        # Pixel format is BIP
        for i in rows:
            for j in cols:
                if allBands:
                    f.seek(offset + i * d_row + j * d_col, 0)
                    vals.fromfile(f, nSubBands)
                else:
                    for k in bands:
                        f.seek(offset +
                               i * d_row +
                               j * d_col +
                               k * d_band, 0)
                        vals.fromfile(f, 1)

        if self.swap:
            vals.byteswap()
        arr = Numeric.array(vals.tolist())
        arr = Numeric.reshape(arr, (nSubRows, nSubCols, nSubBands))

	if self.scale_factor != 1:
	    return arr / float(self.scale_factor)
        return arr

    def read_datum(self, i, j, k):
        '''Reads the band `k` value for pixel at row `i` and column `j`.
	
	Arguments:
	
	    `i`, `j`, `k` (integer):
	    
		Row, column and band index, respectively.
	
	Using this function is not an efficient way to iterate over bands or
	pixels. For such cases, use readBands or readPixel instead.	
	'''
        from array import array

	if self.memmap != None:
	    datum = self.memmap[i, j, k]
	    if self.swap:
		datum = datum.byteswap()
	    if self.scale_factor != 1:
		datum /= float(self.scale_factor)
	    return datum

        vals = array(self.format)
        f = self.fid        
        f.seek(self.offset + self.sample_size \
               * (self.nbands * (i * self.ncols + j) + k), 0)
        # Pixel format is BIP so read entire pixel.
        vals.fromfile(f, 1)
        if self.swap:
            vals.byteswap()
	return vals.tolist()[0] / float(self.scale_factor)

        
        
        
