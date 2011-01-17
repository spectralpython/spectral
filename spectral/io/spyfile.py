#########################################################################
#
#   spyfile.py - This file is part of the Spectral Python (SPy) package.
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
:class:`~spectral.SpyFile` is the base class for creating objects to read
hyperspectral data files.  When a :class:`~spectral.SpyFile` object is created,
it provides an interface to read data from a corresponding file.  When an image
is opened, the actual object returned will be a subclass of :class:`~spectral.SpyFile`
(BipFile, BilFile, or BsqFile) corresponding to the interleave of the data
within the image file.

Let's open our sample image.

    >>> from spectral import *
    >>> img = image('92AV3C')
    >>> img.__class__
    <class spectral.io.bilfile.BilFile at 0x1021ed3b0>
    >>> print img
	    Data Source:   '/Users/thomas/spectral_data/92AV3C'
	    # Rows:            145
	    # Samples:         145
	    # Bands:           220
	    Interleave:        BIL
	    Quantization:  16 bits
	    Data format:         h

The image was not located in the working directory but it was still opened
because it was in a directory specified by the *SPECTRAL_DATA* environment
variable.  Because the image pixel data are interleaved by line, the *image* function returned
a *BilFile* instance.

Since hyperspectral image files can be quite large, only
metadata are read from the file when the :class:`~spectral.SpyFile` object is
first created. Image data values are only read when specifically requested via
:class:`~spectral.SpyFile` methods.  The :class:`~spectral.SpyFile` class
provides a subscript operator that behaves much like the numpy array subscript
operator. The :class:`~spectral.SpyFile` object is subscripted as an *MxNxB*
array where *M* is the number of rows in the image, *N* is the number of
columns, and *B* is thenumber of bands.

    >>> img.shape
    (145, 145, 220)
    >>> pixel = img[50,100]
    >>> pixel.shape
    (220,)
    >>> band6 = img[:,:,5]
    >>> band6.shape
    (145, 145, 1)

The image data values were not read from the file until the subscript operator
calls were performed.  Note that since Python indices start at 0, ``img[50,100]``
refers to the pixel at 51st row and 101st column of the image.  Similarly,
``img[:,:,5]`` refers to all the rows and columns for the 6th band of the image.

:class:`~spectral.SpyFile` subclass instances returned for particular image
files will also provide the following methods:

=============   ================================================================
   Method				Description
=============   ================================================================
readBand	Reads a single band into an *MxN* array
readBands	Reads multiple bands into an *MxNxC* array
readPixel	Reads a single pixel into a length *B* array
readSubRegion	Reads multiple bands from a rectangular sub-region of the image
readSubImage	Reads specified rows, columns, and bands
=============   ================================================================

:class:`~spectral.SpyFile` objects have a ``bands`` member, which is an instance
of a :class:`~spectral.BandInfo` object that contains optional information about
the images spectral bands.
'''

import numpy
from spectral.spectral import Image

def findFilePath(filename):
    '''
    Search cwd and SPECTRAL_DATA directories for the given file.
    '''
    import os
    pathname = None
    dirs = ['.']
    if os.environ.has_key('SPECTRAL_DATA'):
        dirs += os.environ['SPECTRAL_DATA'].split(':')
    for d in dirs:
        testpath = os.path.join(d, filename)
        if os.path.isfile(testpath):
            pathname = testpath
            break
    if not pathname:
        raise IOError('Unable to locate file %s' % filename)
    return pathname

class SpyFile(Image):
    '''A base class for accessing spectral image files'''

    def __init__(self, params, metadata = None):
        from spectral import Image
        Image.__init__(self, params, metadata)
	self.scaleFactor = 1.0		# Number by which to divide values read from file.

    def setParams(self, params, metadata):
        import spectral
        import array
        from exceptions import Exception

        spectral.Image.setParams(self, params, metadata)

        try:
            self.fileName = params.fileName
            self.format = params.format
            self._typecode = params.typecode         # for Numeric module
            self.offset = params.offset
            self.byteOrder = params.byteOrder
            if spectral.byteOrder != self.byteOrder:
                self.swap = 1
            else:
                self.swap = 0
            self.sampleSize = array.array(self.format).itemsize

            self.fid = open(findFilePath(self.fileName), "rb")

            # So that we can use this more like a Numeric array
            self.shape = (self.nRows, self.nCols, self.nBands)
        
        except:
            raise

    def __str__(self):
	'''Prints basic parameters of the associated file.'''
	import spectral as spy
        s =  '\tData Source:   \'%s\'\n' % self.fileName
        s += '\t# Rows:         %6d\n' % (self.nRows)
        s += '\t# Samples:      %6d\n' % (self.nCols)
        s += '\t# Bands:        %6d\n' % (self.shape[2])
	if self.interleave == spy.BIL:
	    interleave = 'BIL'
	elif self.interleave == spy.BIP:
	    interleave = 'BIP'
	else:
	    interleave = 'BSQ'
	s += '\tInterleave:     %6s\n' % (interleave)
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


    def typecode(self):
        '''Returns the typecode of the Numeric array type for the image file.'''
        return self._typecode
    
    def load(self):
	'''Loads the entire image into memory in a :class:`spectral.ImageArray` object.
	
	:class:`spectral.ImageArray` is derived from both :class:`spectral.Image`
	and :class:`numpy.ndarray` so it supports the full :class:`numpy.ndarray`
	interface.  The returns object will have shape `(M,N,B)`, where `M`, `N`,
	and `B` are the numbers of rows, columns, and bands in the image.
	'''
        import spectral
        from spectral.spectral import ImageArray
        from array import array
        
        data = array(self.typecode())
        self.fid.seek(self.offset)
        data.fromfile(self.fid, self.nRows * self.nCols * self.nBands)
        if self.swap:
            data.byteswap()
        npArray = numpy.array(data, ImageArray.format)
        if self.interleave == spectral.BIL:
            npArray.shape = (self.nRows, self.nBands, self.nCols)
            npArray = npArray.transpose([0, 2, 1])
        elif self.interleave == spectral.BSQ:
            npArray.shape = (self.nBands, self.nRows, self.nCols)
            npArray = npArray.transpose([1, 2, 0])
	else:
	    npArray.shape = (self.nRows, self.nCols, self.nBands)
	    
	if self.scaleFactor != 1:
	    npArray /= self.scaleFactor
	
        return ImageArray(npArray, self)

    def __getitem__(self, args):
	'''Subscripting operator that provides a numpy-like interface.
	Usage::
	
	    x = img[i, j]
	    x = img[i, j, k]
	    
	Arguments:
	
	    `i`, `j`, `k` (int or :class:`slice` object)
	    
		Integer subscript indices or slice objects.
	
	The subscript operator emulates the :class:`numpy.ndarray` subscript
	operator, except data are read from the corresponding image file instead
	of an array object in memory.  For frequent access or when accessing
	a large fraction of the image data, consider calling
	:meth:`spectral.SpyFile.load` to load the data into an
	:meth:`spectral.ImageArray` object and using its subscript operator
	instead.
	
	Examples:
	
	    Read the pixel at the 30th row and 51st column of the image::
	    
		pixel = img[29, 50]
		
	    Read the 10th band::
	    
		band = img[:, :, 9]
	    
	    Read the first 30 bands for a square sub-region of the image::
	    
		region = img[50:100, 50:100, :30]
	'''

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
	from spectral import Image

        p = Image.params(self)

        p.fileName = self.fileName
        p.offset = self.offset
        p.byteOrder = self.byteOrder
        p.sampleSize = self.sampleSize

        return p

    def __del__(self):
        self.fid.close()


class SubImage(SpyFile):
    '''
    Represents a rectangular sub-region of a larger SpyFile object.
    '''
    def __init__(self, image, rowRange, colRange):
	'''Creates a :class:`Spectral.SubImage` object for a rectangular sub-region.
	
	Arguments:
	
	    `image` (SpyFile):
	    
		The image for which to define the sub-image.
		
	    `rowRange` (2-tuple):
	    
		Integers [i, j) defining the row limits of the sub-region.

	    `colRange` (2-tuple):
	    
		Integers [i, j) defining the col limits of the sub-region.
	
	Returns:
	
	    A :class:`spectral.SubImage` object providing a :class:`spectral.SpyFile`
	    interface to a sub-region of the image.
	
	Raises:
	
	    :class:`IndexError`
	
	Row and column ranges must be 2-tuples (i,j) where i >= 0 and i < j.

	'''

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
        '''Reads a single band from the image.
	
	Arguments:
	
	    `band` (int):
	    
		Index of band to read.
	
	Returns:
	
	   :class:`numpy.ndarray`
	   
		An `MxN` array of values for the specified band.
	'''
        return self.parent.readSubRegion([self.rowOffset, \
                                self.rowOffset + self.nRows - 1], \
                               [self.colOffset, \
                                self.colOffset + self.nCols - 1], \
                               [band])

    def readBands(self, bands):
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
        return self.parent.readSubRegion([self.rowOffset, \
                                self.rowOffset + self.nRows - 1], \
                               [self.colOffset, \
                                self.colOffset + self.nCols - 1], \
                               bands)

    def readPixel(self, row, col):
        '''Reads the pixel at position (row,col) from the file.
	
	Arguments:
	
	    `row`, `col` (int):
	    
		Indices of the row & column for the pixel
	
	Returns:
	
	   :class:`numpy.ndarray`
	   
		A length-`B` array, where `B` is the number of bands in the image.
	'''
        return self.parent.readPixel(row + self.rowOffset, \
                                col + self.colOffset)

    def readSubImage(self, rows, cols, bands = []):
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
        return self.parent.readSubImage(list(array(rows) + self.rowOffset), \
                                        list(array(cols) + self.colOffset), \
                                        bands)

    def readSubRegion(self, rowBounds, colBounds, bands = None):
        '''
        Reads a contiguous rectangular sub-region from the image.
	
	Arguments:
	
	    `rowBounds` (2-tuple of ints):
	    
		(a, b) -> Rows a through b-1 will be read.
	
	    `colBounds` (2-tuple of ints):
	    
		(a, b) -> Columnss a through b-1 will be read.
	    
	    `bands` (list of ints):
	    
		Optional list of bands to read.  If not specified, all bands
		are read.
	
	Returns:
	
	   :class:`numpy.ndarray`
	   
		An `MxNxL` array.
        '''
        return self.parent.readSubImage(list(array(rowBounds) + self.rowOffset), \
                                        list(array(colBounds) + self.colOffset), \
                                        bands)

def transformImage(matrix, img):
    '''Applies a linear transform to an image.
    
    Arguments:
    
	`matrix` (ndarray):
	
	    The `CxB` linear transform to apply.
	
	`img` (ndarray or :class:`spectral.SpyFile`):
	
	    The `MxNxB` image to be transformed.
    
    Returns (ndarray or :class:spectral.spyfile.TransformedImage`):
    
	The transformed image.
    
    If `img` is an ndarray, then a `MxNxC` ndarray is returned.  If `img` is
    a :class:`spectral.SpyFile`, then a :class:`spectral.spyfile.TransformedImage` is
    returned.
    
    If `img` is an ndarr
    
    '''
    import numpy as np
    if isinstance(img, np.ndarray):
	print 'It is an array'
	ret = np.empty(img.shape[:2] + (matrix.shape[0],), img.dtype)
	for i in range(img.shape[0]):
	    for j in range(img.shape[1]):
		ret[i, j] = np.dot(matrix, img[i, j])
	return ret
    else:
	return TransformedImage(matrix, img)
    
class TransformedImage(Image):
    '''
    An image with a linear transformation applied to each pixel spectrum.
    The transformation is not applied until data is read from the image file.
    '''
    _typecode = 'f'
    
    def __init__(self, matrix, img):
        import numpy.oldnumeric as Numeric

        if not isinstance(img, Image):
            raise Exception('Invalid image argument to to TransformedImage constructor.')

        arrayType = type(Numeric.array([1]))
        if type(matrix) != arrayType:
            raise Exception('First argument must be a transformation matrix.')
        if len(matrix.shape) != 2:
            raise Exception('Transformation matrix has invalid shape.')

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

        orig = self.image.__getitem__(args[:2])
        if len(orig.shape) == 1:
            orig = orig[NewAxis, NewAxis, :]
        elif len(orig.shape) == 2:
            orig = orig[NewAxis, :]
        transformed_xy = zeros([orig.shape[0], orig.shape[1], self.shape[2]], self._typecode)
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
                   
    def load(self):
	'''Loads all the image data, transforms it, and returns it in a numpy array).'''
	data = self.image.load()
	xdata = numpy.empty(self.shape, 'f')
	for i in xrange(self.shape[0]):
	    for j in xrange(self.shape[1]):
		xdata[i, j] = numpy.dot(self.matrix, data[i, j])
	return xdata
	
    def readSubRegion(self, rowBounds, colBounds, bands = None):
        '''
        Reads a contiguous rectangular sub-region from the image. First
        arg is a 2-tuple specifying min and max row indices.  Second arg
        specifies column min and max.  If third argument containing list
        of band indices is not given, all bands are read.
        '''
        from numpy import zeros, dot
        orig = self.image.readSubRegion(rowBounds, colBounds)
        transformed = zeros([orig.shape[0], orig.shape[1], self.shape[2]], self._typecode)
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
        transformed = zeros([orig.shape[0], orig.shape[1], self.shape[2]], self._typecode)
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
    
    def typecode(self):
	return self._typecode
        
def typecode(obj):
    '''
    The typecode method was removed from arrays in the transition from Numeric/Numarray
    to NumPy.  This function returns the appropriate typecode for numpy arrays or
    any object with a typecode() method.
    '''
    import numpy
    if isinstance(obj, numpy.ndarray):
	return obj.dtype.char
    else:
	return obj.typecode()
