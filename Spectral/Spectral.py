#########################################################################
#
#   Spectral.py - This file is part of the Spectral Python (SPy) package.
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

class SpySettings:
    def __init__(self):
	self.viewer = None
	self.plotter = None

settings = SpySettings()

# Default color table
spyColors = numpy.array([[  0,   0,   0],
                   [255,   0,   0],
                   [  0, 255,   0],
                   [  0,   0, 255],
                   [255, 255,   0],
                   [255,   0, 255],
                   [  0, 255, 255],
                   [255, 170,   0],
                   [ 50, 170,   0],
                   [170, 170,   0],
                   [170,   0, 255],
                   [200, 200, 200],
                   [  0, 200, 200],
                   [170,   0, 170],
                   [100, 175, 255],
                   [255, 175, 175],
                   [255, 175, 255],
                   [255, 255, 255],
                   [255, 255, 255],
                   [255, 255, 255],
                   [255, 255, 255],
                   [255, 255, 255],
                   [255, 255, 255],
                   [255, 255, 255],
                   [255, 255, 255],
                   [255, 255, 255],
                   [255, 255, 255],
                   [255, 255, 255],
                   [255, 255, 255],
                   [255, 255, 255]], numpy.int)

class BandInfo:
    '''Data characterizing the spectral bands associated with an image.'''
    def __init__(self):
	self.centers = None
	self.bandwidths = None
	self.centersStdDevs = None
	self.bandwidthsStdDevs = None
	self.bandQuantity = ""
	self.bandUnit = ""

class Image:
    '''Spectral.Image is the common base class for Spectral image objects.'''

    def __init__(self, params, metadata = None):
	self.bands = BandInfo()
        self.setParams(params, metadata)

    def setParams(self, params, metadata):
        import Spectral
        import array
        from exceptions import Exception
        
        try:
            self.nBands = params.nBands
            self.nRows = params.nRows
            self.nCols = params.nCols
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

        p.nBands = self.nBands
        p.nRows = self.nRows
        p.nCols = self.nCols
        p.format = self.format
        p.metadata = self.metadata
        p.typecode = self._typecode

        return p

    def __repr__(self):
        return self.__str__()
       
class ImageArray(numpy.ndarray, Image):

    format = 'f'	# Use 4-byte floats form data arrays
    
    def __new__(subclass, data, spyFile):
        from Io.SpyFile import SpyFile
        
        obj = numpy.asarray(data).view(ImageArray)
        # Add param data to Image initializer
        Image.__init__(obj, spyFile.params(), spyFile.metadata)
	obj.bands = spyFile.bands
        return obj

    def typecode(self):
        return self.dtype.char

    def __repr__(self):
        return self.__str__()
    
    def readBand(self, i):
	'''For compatibility with SpyFile objects. Returns arr[:,:,i]'''
	return self[:, :, i]
    
    def readBands(self, bands):
	'''For compatibility with SpyFile objects. Equivlalent to arr.take(bands, 2)'''
	return self.take(bands, 2)    

    def readPixel(self, row, col):
	'''For compatibility with SpyFile objects. Equivlalent to arr[row, col]'''
	return self[row, col]

    def readDatum(self, i, j, k):
	'''For compatibility with SpyFile objects. Equivlalent to arr[i, j, k]'''
	return self[i, j, k]
    
    def load(self):
	'''For compatibility with SpyFile objects. Returns self'''
	return self
    
    def info(self):
        s = '\t# Rows:         %6d\n' % (self.nRows)
        s += '\t# Samples:      %6d\n' % (self.nCols)
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

def image(file):
    '''
    Try to locate and determine the type of an image file and open it.

    USAGE: im = image(file)
    '''

    from exceptions import IOError
    import os
    from Io import Aviris, Envi, Erdas, SpyFile
    from Io.SpyFile import findFilePath

    pathname = findFilePath(file)
        
    # Try to open it as an ENVI header file.
    try:
        return Envi.open(pathname)
    except:
        pass

    # Maybe it's an Erdas Lan file
    try:
        return Erdas.open(pathname)
    except:
        pass

    # See if the size is consistent with an Aviris file
    try:
        return Aviris.open(pathname)
    except:
        pass

    raise IOError, 'Unable to determine file type or type not supported.'

def tileImage(im, nRows, nCols):
    '''
    Break an image into nRows x nCols tiles.

    USAGE: tiles = tileImage(im, nRows, nCols)

    ARGUMENTS:
        im              The SpyFile to tile.
        nRows           Number of tiles in the veritical direction.
        nCols           Number of tiles in the horizontal direction.

    RETURN VALUE:
        tiles           A list of lists of SubImage objects. tiles
                        contains nRows lists, each of which contains
                        nCols SubImage objects.
    '''

    from numpy.oldnumeric import array, Int
    from Io.SpyFile import SubImage
    x = (array(range(nRows + 1)) * float(im.nRows) / nRows).astype(Int)
    y = (array(range(nCols + 1)) * float(im.nCols) / nCols).astype(Int)
    x[-1] = im.nRows
    y[-1] = im.nCols

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

def saveTrainingSets(sets, file):
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
    
def loadTrainingSets(file, im = 0):
    '''
    Loads a list of TrainingSet objects from a file.  This function assumes
    that all the sets in the list refer to the same image and mask array.
    If that is not the case, this function should not be used.
    '''
    from Spectral.Algorithms import TrainingSet
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



