#########################################################################
#
#   SpyFile.py - This file is part of the Spectral Python (SPy) package.
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


from Numeric import *

class SpyFile:
    '''A base class for accessing spectral image files'''

    def __init__(self, params, metadata = None):
        import Spectral
        import array
        from exceptions import Exception

        try:
        
            self.fileName = params.fileName
            self.nBands = params.nBands
            self.nRows = params.nRows
            self.nCols = params.nCols
            self.format = params.format
            self._typecode = params.typecode         # for Numeric module
            self.offset = params.offset
            self.byteOrder = params.byteOrder
            if Spectral.byteOrder != self.byteOrder:
                self.swap = 1
            else:
                self.swap = 0
            self.sampleSize = array.array(self.format).itemsize

            if not metadata:
                self.metadata = {}
            else:
                self.metadata = metadata

            self.fid = open(self.fileName, "rb")

            # So that we can use this more like a Numeric array
            self.shape = (self.nRows, self.nCols, self.nBands)
        
        except:
            raise

    def typecode(self):
        '''Returns the typecode of the Numeric array type for this
        image file.
        '''
        return self._typecode
    

    def __getitem__(self, args):

        intType = type(1)
        sliceType = type(slice(0,0,0))

        if len(args) < 2:
            raise IndexError, 'Too few subscript indices.'

        if type(args[0]) == intType and type(args[1]) == intType \
           and len(args) == 2:
            return self.readPixel(args[0], args[1])
        elif len(args) == 3 and (type(args[0]) == intType \
                                 and type(args[1]) == intType \
                                 and type(args[2]) == intType):
            return self.readDatum(args[0], args[1], args[2])
        else:
            #  At least one arg should be a slice
            if type(args[0]) == sliceType:
                (xstart, xstop, xstep) = (args[0].start, args[0].stop, \
                                          args[0].step)
                if xstart == None:
                    xstart = 0;
                if xstop == None:
                    xstop = self.nRows
                if xstep == None:
                    xstep = 1
                rows = range(xstart, xstop, xstep)
            else:
                rows = [args[0]]
            if type(args[1]) == sliceType:
                (ystart, ystop, ystep) = (args[1].start, args[1].stop, \
                                          args[1].step)
                if ystart == None:
                    ystart = 0;
                if ystop == None:
                    ystop = self.nCols
                if ystep == None:
                    ystep = 1
                cols = range(ystart, ystop, ystep)
            else:
                cols = [args[1]]

        if len(args) == 2:
            bands = range(self.nBands)
        elif type(args[2]) == sliceType:
            (zstart, zstop, zstep) = (args[2].start, args[2].stop, \
                                      args[2].step)
            if zstart == None:
                zstart = 0
            if zstop == None:
                zstop = self.nBands
            if zstep == None:
                zstep = 1
            bands = range(zstart, zstop, zstep)
        elif type(args[2] == intType):
            bands = [args[2]]
        else:
            bands = args[2]
            
        return self.readSubImage(rows, cols, bands)



    def params(self):
        '''Return an object containing the SpyFile parameters.'''

        class P: pass
        p = P()

        p.fileName = self.fileName
        p.nBands = self.nBands
        p.nRows = self.nRows
        p.nCols = self.nCols
        p.format = self.format
        p.offset = self.offset
        p.byteOrder = self.byteOrder
        p.swap = self.swap
        p.sampleSize = self.sampleSize
        p.metadata = self.metadata
        p.typecode = self._typecode

        return p

    def __del__(self):
        self.fid.close()
        


class SubImage(SpyFile):
    '''
    Represents a rectangular sub-region of a larger SpyFile object.
    '''

    def __init__(self, image, rowRange, colRange):

        import exceptions

        if rowRange[0] < 0 or \
           rowRange[1] > image.nRows or \
           colRange[0] < 0 or \
           colRange[1] > image.nCols:
            raise IndexError, 'SubImage index out of range.'

        p = image.params()
        p.nRows = rowRange[1] - rowRange[0]
        p.nCols = colRange[1] - colRange[0]

        SpyFile.__init__(self, p, image.metadata)
        self.parent = image
        self.rowOffset = rowRange[0]
        self.colOffset = colRange[0]

    def readBand(self, band):
        return self.parent.readSubRegion([self.rowOffset, \
                                self.rowOffset + self.nRows - 1], \
                               [self.colOffset, \
                                self.colOffset + self.nCols - 1], \
                               [band])

    def readBands(self, bands):
        return self.parent.readSubRegion([self.rowOffset, \
                                self.rowOffset + self.nRows - 1], \
                               [self.colOffset, \
                                self.colOffset + self.nCols - 1], \
                               bands)

    def readPixel(self, row, col):
        return self.parent.readPixel(row + self.rowOffset, \
                                col + self.colOffset)

    def readSubImage(self, rowBounds, colBounds, bands = []):
        return self.parent.readSubImage([rowBounds[0] + self.rowOffset, \
                                    rowBounds[1] + self.rowOffset], \
                                   [colBounds[0] + self.colOffset, \
                                    colBounds[1] + self.colOffset], \
                                   bands)

