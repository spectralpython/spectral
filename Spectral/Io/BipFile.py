#########################################################################
#
#   BipFile.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2001 Thomas Boggs
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

from SpyFile import SpyFile

class BipFile(SpyFile):
    '''
    A class to represent image files stored with bands interleaved
    by pixel.
    '''

    def __init__(self, params, metadata = None):
        import Spectral
        self.interleave = Spectral.BIP
        if metadata == None:
            metadata = {}
        SpyFile.__init__(self, params, metadata)        

    def readBand(self, band):
        '''Read a single band from the image.'''

        from array import array
        import Numeric

        vals = array(self.format)
        delta = self.sampleSize * (self.nBands - 1) 
        nVals = self.nRows * self.nCols

        f = self.fid

        f.seek(self.offset + self.sampleSize * band, 0)
        
        # Pixel format is BIP
        for i in range(nVals - 1):
            vals.fromfile(f, 1)
            f.seek(delta, 1)
        vals.fromfile(f, 1)

        if self.swap:
            vals.byteswap()            
        arr = Numeric.array(vals.tolist())
        arr = Numeric.reshape(arr, (self.nRows, self.nCols))

        return arr

    def readBands(self, bands):
        '''Read specified bands from the image.'''

        from array import array
        import Numeric

        vals = array(self.format)
        offset = self.offset
        delta = self.sampleSize * self.nBands
        nVals = self.nRows * self.nCols

        # Increments between bands
        delta_b = bands[:]
        for i in range(len(delta_b)):
            delta_b[i] *= self.sampleSize

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
        arr = Numeric.reshape(arr, (self.nRows, self.nCols, len(bands)))

        return arr

    def readPixel(self, row, col):
        '''Read the pixel at position (row,col) from the file.'''

        from array import array
        import Numeric

        vals = array(self.format)

        f = self.fid        
        f.seek(self.offset + self.sampleSize \
               * self.nBands * (row * self.nCols + col), 0)
        # Pixel format is BIP so read entire pixel.
        vals.fromfile(f, self.nBands)

        if self.swap:
            vals.byteswap()
        pixel = Numeric.array(vals.tolist())

        return pixel

    def readSubRegion(self, rowBounds, colBounds, bands = None):
        '''
        Reads a contiguous rectangular sub-region from the image. First
        arg is a 2-tuple specifying min and max row indices.  Second arg
        specifies column min and max.  If third argument containing list
        of band indices is not given, all bands are read.
        '''

        import array
        import Numeric

        offset = self.offset
        nBands = self.nBands
        nSubRows = rowBounds[1] - rowBounds[0]  # Rows in sub-image
        nSubCols = colBounds[1] - colBounds[0]  # Cols in sub-image
        d_row = self.sampleSize * self.nCols * self.nBands
        colStartPos = colBounds[0] * self.sampleSize * self.nBands
        vals = array.array(self.format)
        nVals = self.nRows * self.nCols

        # Increments between bands
        if bands != None:
            allBands = 0
            nSubBands = len(bands)
            delta_b = bands[:]
            for i in range(len(delta_b)):
                delta_b[i] *= self.sampleSize
        else:
            allBands = 1
            nSubBands = self.nBands

        f = self.fid
        
        # Pixel format is BIP
        for i in range(rowBounds[0], rowBounds[1]):
            f.seek(offset + i * d_row + colStartPos, 0)
            rowPos = f.tell()

            if allBands:
                # This is the simple one
                vals.fromfile(f, nSubCols * nBands)
            else:
                # Need to pull out specific bands for each column.
                for j in range(nSubCols):
                    f.seek(rowPos + j * self.sampleSize * self.nBands, 0)
                    pixelPos = f.tell()
                    for k in range(len(bands)):
                        f.seek(pixelPos + delta_b[k], 0)    # Next band
                        vals.fromfile(f, 1)

        if self.swap:
            vals.byteswap()
        arr = Numeric.array(vals.tolist())
        arr = Numeric.reshape(arr, (nSubRows, nSubCols, nSubBands))

        return arr


    def readSubImage(self, rows, cols, bands = None):
        '''
        Reads a sub-image from a rectangular region within the image.
        First arg is a 2-tuple specifying min and max row indices.
        Second arg specifies column min and max. If third argument
        containing list of band indices is not given, all bands are read.
        '''
        import array
        import Numeric

        offset = self.offset
        nBands = self.nBands
        nSubRows = len(rows)                        # Rows in sub-image
        nSubCols = len(cols)                        # Cols in sub-image
        d_band = self.sampleSize
        d_col = d_band * self.nBands
        d_row = d_col * self.nCols
        vals = array.array(self.format)
        nVals = self.nRows * self.nCols

        # Increments between bands
        if bands != None:
            allBands = 0
            nSubBands = len(bands)
        else:
            allBands = 1
            bands = range(self.nBands)
            nSubBands = self.nBands

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

        return arr

        
        
        
        
