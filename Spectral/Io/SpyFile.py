#########################################################################
#
#   SpyFile.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2001-2008 Thomas Boggs
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
Common code for accessing hyperspectral image files.
'''

import numpy

def findFilePath(filename):
    '''
    Search cwd and SPECTRAL_DATA directories for the given file.
    '''
    import os
    pathname = None
    dirs = ['.']
    if os.environ.has_key('SPECTRAL_DATA'):
        dirs += os.environ['SPECTRAL_DATA'].split(';')
    for d in dirs:
        testpath = os.path.join(d, filename)
        if os.path.isfile(testpath):
            pathname = testpath
            break
    if not pathname:
        raise IOError('Unable to locate file ' % filename)
    return pathname

class SpyFile:
    '''A base class for accessing spectral image files'''

    def __init__(self, params, metadata = None):

        self.setParams(params, metadata)

    def setParams(self, params, metadata):
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

            self.fid = open(findFilePath(self.fileName), "rb")

            # So that we can use this more like a Numeric array
            self.shape = (self.nRows, self.nCols, self.nBands)
        
        except:
            raise

    def __str__(self):
        s =  '\tData Source:   \'%s\'\n' % self.fileName
        s += '\t# Rows:         %6d\n' % (self.nRows)
        s += '\t# Samples:      %6d\n' % (self.nCols)
        s += '\t# Bands:        %6d\n' % (self.shape[2])
        s += '\tQuantization: %3d bits\n' % (self.sampleSize * 8)

        tc = self._typecode
        if tc == '1':
            tcs = 'char'
        elif tc == 's':
            tcs = 'Int16'
        elif tc == 'i':
            tcs = Int32
        elif tc == 'f':
            tcs = 'Float32'
        elif tc == 'd':
            tcs = 'Float64'
        else:
            tcs = self._typecode
            
        s += '\tData format:  %8s' % tcs
        return s

    def __repr__(self):
        return self.__str__()

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

        if len(args) == 2 or args[2] == None:
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
        elif type(args[2]) == intType:
            bands = [args[2]]
        else:
            # Band indices should be in a list
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

        SpyFile.__init__(self, p, image.metadata)
        self.parent = image
        self.rowOffset = rowRange[0]
        self.colOffset = colRange[0]
        self.nRows = rowRange[1] - rowRange[0]
        self.nCols = colRange[1] - colRange[0]
        self.shape = (self.nRows, self.nCols, self.nBands)

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

    def readSubImage(self, rows, cols, bands = []):
        return self.parent.readSubImage(list(array(rows) + self.rowOffset), \
                                        list(array(cols) + self.colOffset), \
                                        bands)

    def readSubRegion(self, rowBounds, colBounds, bands = None):
        '''
        Reads a contiguous rectangular sub-region from the image. First
        arg is a 2-tuple specifying min and max row indices.  Second arg
        specifies column min and max.  If third argument containing list
        of band indices is not given, all bands are read.
        '''
        return self.parent.readSubImage(list(array(rowBounds) + self.rowOffset), \
                                        list(array(colBounds) + self.colOffset), \
                                        bands)

class TransformedImage(SpyFile):
    '''
    An image with a linear transformation applied to each pixel spectrum.
    The transformation is not applied until data is read from the image file.
    '''
    
    def __init__(self, matrix, img):
        import numpy.oldnumeric as Numeric

        if not isinstance(img, SpyFile):
            raise 'Invalid image argument to to TransformedImage constructor.'

        arrayType = type(Numeric.array([1]))
        if type(matrix) != arrayType:
            raise 'First argument must be a transformation matrix.'
        if len(matrix.shape) != 2:
            raise 'Transformation matrix has invalid shape.'

        params = img.params()
        self.setParams(params, params.metadata)


        # If img is also a TransformedImage, then just modify the transform
        if isinstance(img, TransformedImage):
            self.matrix = matrixmultiply(matrix, img.matrix)
            self.image = img.image
            if matrix.shape[1] != img.matrix.shape[0]:
                raise 'Invalid shape for transformation matrix.'
            # Set shape to what it will be after linear transformation
            self.shape = [self.image.shape[0], self.image.shape[1], matrix.shape[0]]
        else:
            self.matrix = matrix
            self.image = img
            if matrix.shape[1] != img.nBands:
                raise 'Invalid shape for transformation matrix.'
            # Set shape to what it will be after linear transformation
            self.shape = [img.shape[0], img.shape[1], matrix.shape[0]]

        self.nBands = matrix.shape[0]

    def __getitem__(self, args):
        '''
        Get data from the image and apply the transform.
        '''
        from numpy import zeros, dot, take
        from numpy.oldnumeric import NewAxis
        if len(args) < 2:
            raise 'Must pass at least two subscript arguments'

        # Note that band indices are wrt transformed features
        if len(args) == 2 or args[2] == None:
            bands = range(self.nBands)
        elif type(args[2]) == slice:
            (zstart, zstop, zstep) = (args[2].start, args[2].stop, \
                                      args[2].step)
            if zstart == None:
                zstart = 0
            if zstop == None:
                zstop = self.nBands
            if zstep == None:
                zstep = 1
            bands = range(zstart, zstop, zstep)
        elif isinstance(args[2], int):
            bands = [args[2]]
        else:
            # Band indices should be in a list
            bands = args[2]

        orig = SpyFile.__getitem__(self.image, args[:2])
        if len(orig.shape) == 1:
            orig = orig[NewAxis, NewAxis, :]
        elif len(orig.shape) == 2:
            orig = orig[NewAxis, :]
        transformed_xy = zeros([orig.shape[0], orig.shape[1], self.shape[2]], float)
        for i in range(transformed_xy.shape[0]):
            for j in range(transformed_xy.shape[1]):
                transformed_xy[i, j] = dot(self.matrix, orig[i, j])
        # Remove unnecessary dimensions

        transformed = take(transformed_xy, bands, 2)
        
        if transformed.shape[0] == 1:
            transformed.shape = transformed.shape[1:]
        if transformed.shape[0] == 1:
            transformed.shape = transformed.shape[1:]
            
        return transformed


    def readPixel(self, row, col):
        return numpy.dot(self.matrix, self.image.readPixel(row, col))                       
                   
    
    def readSubRegion(self, rowBounds, colBounds, bands = None):
        '''
        Reads a contiguous rectangular sub-region from the image. First
        arg is a 2-tuple specifying min and max row indices.  Second arg
        specifies column min and max.  If third argument containing list
        of band indices is not given, all bands are read.
        '''
        from numpy import zeros, dot
        orig = self.image.readSubRegion(rowBounds, colBounds)
        transformed = zeros([orig.shape[0], orig.shape[1], self.shape[2]], float)
        for i in range(transformed.shape[0]):
            for j in range(transformed.shape[1]):
                transformed[i, j] = dot(self.matrix, orig[i, j])
        if bands:
            return numpy.take(transformed, bands, 2)
        else:
            return transformed


    def readSubImage(self, rows, cols, bands = None):
        '''
        Reads a sub-image from a rectangular region within the image.
        First arg is a 2-tuple specifying min and max row indices.
        Second arg specifies column min and max. If third argument
        containing list of band indices is not given, all bands are read.
        '''
        from numpy import zeros, dot
        orig = self.image.readSubImage(rows, cols)
        transformed = zeros([orig.shape[0], orig.shape[1], self.shape[2]], float)
        for i in range(transformed.shape[0]):
            for j in range(transformed.shape[1]):
                transformed[i, j] = dot(self.matrix, orig[i, j])
        if bands:
            return numpy.take(transformed, bands, 2)
        else:
            return transformed

    def readDatum(self, i, j, k):
        return numpy.take(self.readPixel(i, j), k)

    def readBands(self, bands):
        shape = (self.image.nRows, self.image.nCols, self.nBands)
        data = numpy.zeros(shape, float)
        for i in range(shape[0]):
            for j in range(shape[1]):
                data[i, j] = numpy.take(self.readPixel(i, j), bands)
        return data
        
