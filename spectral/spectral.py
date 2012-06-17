#########################################################################
#
#   spectral.py - This file is part of the Spectral Python (SPy) package.
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
Generic functions for handling spectral image files.
'''

import numpy

from exceptions import DeprecationWarning
from warnings import warn

class SpySettings:
    def __init__(self):
	self.viewer = None
	self.plotter = None

settings = SpySettings()

# Default color table
spy_colors = numpy.array([[  0,   0,   0],
                   [255,   0,   0],
                   [  0, 255,   0],
                   [  0,   0, 255],
		   [255, 255,   0],
                   [255,   0, 255],
                   [  0, 255, 255],
                   [200, 100,   0],
                   [  0, 200, 100],
                   [100,   0, 200],
                   [200,   0, 100],
                   [100, 200,   0],
                   [  0, 100, 200],
                   [150,  75,  75],
                   [ 75, 150,  75],
                   [ 75,  75, 150],
                   [255, 100, 100],
                   [100, 255, 100],
                   [100, 100, 255],
                   [255, 150,  75],
                   [ 75, 255, 150],
                   [150,  75, 255],
		   [ 50,  50,  50],
		   [100, 100, 100],
		   [150, 150, 150],
		   [200, 200, 200],
		   [250, 250, 250],
		   [100,   0,   0],
		   [200,   0,   0],
		   [  0, 100,   0],
		   [  0, 200,   0],
		   [  0,   0, 100],
		   [  0,   0, 200],
		   [100, 100,   0],
		   [200, 200,   0],
		   [100,   0, 100],
		   [200,   0, 200],
		   [  0, 100, 100],
		   [  0, 200, 200]], numpy.int)


class BandInfo:
    '''A BandInfo object characterizes the spectral bands associated with an image.
    All BandInfo member variables are optional.  For *N* bands, all members of
    type <list> will have length *N* and contain float values.
    
    =================	===================================== 	=======
        Member			Description		    	Default
    =================	===================================== 	=======
    centers		List of band centers		    	None
    bandwidths		List of band FWHM values	    	None
    centers_stdevs	List of std devs of band centers    	None
    bandwidth_stdevs	List of std devs of bands FWHMs     	None
    band_quantity	Image data type (e.g., "reflectance")	""
    band_unit		Band unit (e.g., "nanometer")	    	""
    =================	===================================== 	=======
    '''
    def __init__(self):
	self.centers = None
	self.bandwidths = None
	self.centers_stdevs = None
	self.bandwidth_stdevs = None
	self.band_quantity = ""
	self.band_unit = ""

class Image():
    '''spectral.Image is the common base class for spectral image objects.'''

    def __init__(self, params, metadata = None):
	self.bands = BandInfo()
        self.set_params(params, metadata)

    def set_params(self, params, metadata):
        import spectral
        import array
        from exceptions import Exception
        
        try:
            self.nbands = params.nbands
            self.nrows = params.nrows
            self.ncols = params.ncols
            self._typecode = params.typecode         # for Numeric module

            if not metadata:
                self.metadata = {}
            else:
                self.metadata = metadata    
        except:
            raise

    def params(self):
        '''Return an object containing the SpyFile parameters.'''

        class P: pass
        p = P()

        p.nbands = self.nbands
        p.nrows = self.nrows
        p.ncols = self.ncols
        p.format = self.format
        p.metadata = self.metadata
        p.typecode = self._typecode

        return p

    def __repr__(self):
        return self.__str__()

    def setParams(self, *args):
	warn('Image.setParams has been deprecated.  Use Image.set_params',
	     DeprecationWarning)
	return self.set_params(*args)
       
class ImageArray(numpy.ndarray, Image):
    '''ImageArray is an interface to an image loaded entirely into memory.
    ImageArray objects are returned by :meth:`spectral.SpyFile.load`.
    This class inherits from both numpy.ndarray and SpyFile, providing the interfaces
    of both classes.
    '''

    format = 'f'	# Use 4-byte floats form data arrays
    
    def __new__(subclass, data, spyFile):
        from io.spyfile import SpyFile
        
        obj = numpy.asarray(data).view(ImageArray)
        # Add param data to Image initializer
        Image.__init__(obj, spyFile.params(), spyFile.metadata)
	obj.bands = spyFile.bands
        return obj

    def typecode(self):
        return self.dtype.char

    def __repr__(self):
        return self.__str__()
    
    def read_band(self, i):
	'''For compatibility with SpyFile objects. Returns arr[:,:,i]'''
	return self[:, :, i]
    
    def read_bands(self, bands):
	'''For compatibility with SpyFile objects. Equivlalent to arr.take(bands, 2)'''
	return self.take(bands, 2)    

    def read_pixel(self, row, col):
	'''For compatibility with SpyFile objects. Equivlalent to arr[row, col]'''
	return self[row, col]

    def read_datum(self, i, j, k):
	'''For compatibility with SpyFile objects. Equivlalent to arr[i, j, k]'''
	return self[i, j, k]
    
    def load(self):
	'''For compatibility with SpyFile objects. Returns self'''
	return self
    
    def __getitem__(self, key):
	import numpy
	return numpy.array(numpy.ndarray.__getitem__(self, key))
    
    def info(self):
        s = '\t# Rows:         %6d\n' % (self.nrows)
        s += '\t# Samples:      %6d\n' % (self.ncols)
        s += '\t# Bands:        %6d\n' % (self.shape[2])

        tc = self.typecode()
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
            tcs = 'unknown'
            
        s += '\tData format:  %8s' % tcs
        return s

    # Deprecated methods
    def readBand(self, i):
	warn('ImageArray.readBand has been deprecated.  Use ImageArray.read_band.',
	     DeprecationWarning)
	return self.read_band(i)
    def readBands(self, bands):
	warn('ImageArray.readBands has been deprecated.  Use ImageArray.read_bands.',
	     DeprecationWarning)
	return self.read_bands(bands)
    def readPixel(self, row, col):
	warn('ImageArray.readPixel has been deprecated.  Use ImageArray.read_pixel.',
	     DeprecationWarning)
	return self.read_pixel(bands)
    def readDatum(self, i, j, k):
	warn('ImageArray.readDatum has been deprecated.  Use ImageArray.read_datum.',
	     DeprecationWarning)
	return self.read_datum(i, j, k)

def image(file):
    '''
    Locates & opens the specified hyperspectral image.

    Arguments:
    
	file (str):N
	    Name of the file to open.
	
    Returns:
    
	SpyFile object to access the file.
    
    Raises:
    
	IOError.
	
    This function attempts to determine the associated file type and open the file.
    If the specified file is not found in the current directory, all directories
    listed in the :const:`SPECTRAL_DATA` environment variable will be searched
    until the file is found.  If the file being opened is an ENVI file, the
    `file` argument should be the name of the header file.
    '''

    from exceptions import IOError
    import os
    from io import aviris, envi, erdas, spyfile
    from io.spyfile import find_file_path

    pathname = find_file_path(file)
        
    # Try to open it as an ENVI header file.
    try:
        return envi.open(pathname)
    except:
        pass

    # Maybe it's an Erdas Lan file
    try:
        return erdas.open(pathname)
    except:
        pass

    # See if the size is consistent with an Aviris file
    try:
        return aviris.open(pathname)
    except:
        pass

    raise IOError, 'Unable to determine file type or type not supported.'

def tile_image(im, nrows, ncols):
    '''
    Break an image into nrows x ncols tiles.

    USAGE: tiles = tile_image(im, nrows, ncols)

    ARGUMENTS:
        im              The SpyFile to tile.
        nrows           Number of tiles in the veritical direction.
        ncols           Number of tiles in the horizontal direction.

    RETURN VALUE:
        tiles           A list of lists of SubImage objects. tiles
                        contains nrows lists, each of which contains
                        ncols SubImage objects.
    '''

    from numpy.oldnumeric import array, Int
    from io.spyfile import SubImage
    x = (array(range(nrows + 1)) * float(im.nrows) / nrows).astype(Int)
    y = (array(range(ncols + 1)) * float(im.ncols) / ncols).astype(Int)
    x[-1] = im.nrows
    y[-1] = im.ncols

    tiles = []
    for r in range(len(x) - 1):
        row = []
        for c in range(len(y) - 1):
            si = SubImage(im, [x[r], x[r + 1]], [y[c], y[c + 1]])
            row.append(si)
        tiles.append(row)
    return tiles


def help(x):
    '''
    Prints the __doc__ string for x.

    USAGE: help(x)
    
    Note that this produces the same output as 'print x.__doc__'.
    '''

    # TO DO:
    # If the arg is a package/module name, make this also print out
    # the names of its sub-items.
    
    print x.__doc__

def save_training_sets(sets, file):
    '''
    Saves a list of TrainingSet objects to a file.  This function assumes
    that all the sets in the list refer to the same image and mask array.
    If that is not the case, this function should not be used.
    '''
    import pickle
    
    f = open(file, 'w')
    z = array([])
    
    pickle.dump(len(sets), f)
    DumpArray(sets[0].mask, f)
    for s in sets:
        s.mask = z
        s.dump(f)

    f.close()
    
def load_training_sets(file, im = 0):
    '''
    Loads a list of TrainingSet objects from a file.  This function assumes
    that all the sets in the list refer to the same image and mask array.
    If that is not the case, this function should not be used.
    '''
    from spectral.algorithms import TrainingSet
    import pickle
    
    f = open(file)
    sets = []

    nSets = pickle.load(f)
    mask = LoadArray(f)
    for i in range(nSets):
        s = TrainingSet(0,0)
        s.load(f)
        s.mask = mask
        if im:
            s.image = im
        else:
            s.image = image(s.image)
        sets.append(s)

    f.close()
    return sets   

# Deprecated Functions

def tileImage(im, nrows, ncols):
    warn('tile_image has been deprecated.  Use tile_image.',
	 DeprecationWarning)
    return tile_image(im, nrows, ncols)

def saveTrainingSets(sets, file):
    warn('save_training_sets has been deprecated.  Use save_training_sets.',
	 DeprecationWarning)
    return save_training_sets(sets, file)

def loadTrainingSets(file, im = 0):
    warn('load_training_sets has been deprecated.  Use load_training_sets.',
	 DeprecationWarning)
    return load_training_sets(file, im)
