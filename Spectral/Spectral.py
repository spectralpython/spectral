#########################################################################
#
#   Spectral.py - This file is part of the Spectral Python (SPy) package.
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
Generic functions for handling spectral image files.
'''

from Numeric import *

# Default color table
spyColors = array([[  0,   0,   0],
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
                   [255, 255, 255]], Int)


def image(file):
    '''
    Try to automatically determine the type of an image file and open it.

    USAGE: im = image(file)
    '''

    from exceptions import IOError
    import os

    # Raise an exception if the file does not exist.
    s = os.stat(file)

    # Try to open it as an ENVI header file.
    from Io.Envi import EnviHdr
    try:
        return EnviHdr(file)
    except:
        pass

    # Maybe it's an Erdas Lan file
    from Io.Erdas import *
    try:
        return ErdasLan(file)
    except:
        pass

    # See if the size is consistent with an Aviris file
    from Io.Aviris import Aviris
    try:
        return Aviris(file)
    except:
        pass

    raise IOError, 'Unable to determine file type or type not supported.'


def initWxPython():
    '''Use wxPython for image display.'''
    global viewer
    import Graphics.SpyWxPython
    viewer = Graphics.SpyWxPython
    viewer.init()


def initNumTut():
    '''Use NumTut for image display.'''
    global viewer
    import Graphics.SpyNumTut
    viewer = Graphics.SpyNumTut

def initGraphics():
    '''Initialize default graphics handlers.'''

    try:
        initWxPython()
    except:
        print 'Unable to initialize wxWindows.'
        try:
            initNumTut()
        except:
            print 'Unable to initialize NumTut.'
            print 'No viewers initialized.'


def view(*args, **kwargs):
    '''
    Open a window and display an RGB image.

    USAGE: view(source [, bands] [stretch = 1] [stretchAll = 1]
                [bounds = (lower, upper)] )

    source is the data source and can be either a SpyFile object or a
    NumPy array.  bands is an optional list which specifies the RGB
    channels to display. If bands is not present and source is a SpyFile
    object, it's metadata dict will be checked if it contains a "default
    bands" item.  Otherwise, the first, middle and last band will be
    displayed. If stretch is defined, the image data will be scaled
    so that the maximum value in the display data will be 1. If
    stretchAll is defined, each color channel will be scaled separately
    so that its maximum value is 1. If bounds is specified, the data will
    be scaled so that lower and upper correspond to 0 and 1, respectively
    . Any values outside of the range (lower, upper) will be clipped.
    '''
    apply(viewer.view, args, kwargs)


def viewIndexed(*args, **kwargs):
    '''
    Open a window and display an indexed color image.

    USAGE: viewIndexed(source [, colors])

    source is the data source and can be either a SpyFile object or a
    NumPy array. The optional argument colors is an Nx3 NumPy array
    which specifies the RGB colors for the color indices in source.
    Each column of colors specifies the red, green, and blue color
    components in the range [0, 255]. If colors is not specified, the
    default color table is used.
    '''

    from Spectral import *

    if not kwargs.has_key('colors'):
        kwargs['colors'] = spyColors
    apply(viewer.view, args, kwargs)
    

def saveImage(*args, **kwargs):
    '''
    Save data as a JPEG image file.

    USAGE: view(file, source [, bands] [stretch = 1] [stretchAll = 1]
                [bounds = (lower, upper)] )

    source is the data source and can be either a SpyFile object or a
    NumPy array.  bands is an optional list which specifies the RGB
    channels to display. If bands is not present and source is a SpyFile
    object, it's metadata dict will be checked if it contains a "default
    bands" item.  Otherwise, the first, middle and last band will be
    displayed. If stretch is defined, the image data will be scaled
    so that the maximum value in the display data will be 1. If
    stretchAll is defined, each color channel will be scaled separately
    so that its maximum value is 1. If bounds is specified, the data will
    be scaled so that lower and upper correspond to 0 and 1, respectively
    . Any values outside of the range (lower, upper) will be clipped.
    '''

    from Io import *
    import Graphics
    from Numeric import *
    import StringIO
    import Image, ImageDraw

    rgb = apply(Graphics.getImageDisplayData, args[1:], kwargs)

    if not kwargs.has_key("colors"):
        rgb = (rgb * 255).astype(UnsignedInt8)
    else:
        rgb = rgb.astype(UnsignedInt8)
    rgb = transpose(rgb, (1, 0, 2))
    im = Image.new("RGB", rgb.shape[:2])
    draw = ImageDraw.ImageDraw(im)

    # TO DO:
    # Find a more efficient way to write data to the PIL image below.
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            draw.point((i, j), tuple(rgb[i, j]))
#    im.putdata(rgb.flat)

    if kwargs.has_key("format"):
        fmt = kwargs["format"]
    else:
        fmt = "JPEG"
        
    im.save(args[0], fmt, quality = 100)    
    

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

    from Numeric import array, Int
    x = (array(range(nRows + 1)) * float(im.nRows) / nRows).astype(Int)
    y = (array(range(nCols + 1)) * float(im.nCols) / nCols).astype(Int)
    x[-1] = im.nRows
    y[-1] = im.nCols
    print x
    print y

    tiles = []
    for r in range(len(x) - 1):
        row = []
        for c in range(len(y) - 1):
            si = SubRegion(im, [x[r], x[r + 1]], [y[c], y[c + 1]])
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



