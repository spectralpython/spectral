#########################################################################
#
#   Graphics.py - This file is part of the Spectral Python (SPy) package.
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
Common functions for extracting and manipulating data for graphical
display.
'''

def getImageDisplayData(source, bands = None, **kwargs):
    '''
    Extract RGB data to be displayed from a SpyImage or NumPy array.

    USAGE: rgb = getImageDisplayData(source [, bands] [stretch = 1]
                    [stretchAll = 1] [bounds = (lower, upper)] )

    source is the data source and can be either a SpyFile object or a
    NumPy array.  bands is an optional list which specifies the RGB
    channels to display. If bands is not present and source is a SpyFile
    object, it's metadata dict will be checked if it contains a "default
    bands" item.  Otherwise, the first, middle and last band will be
    displayed. If stretch is defined, the contents of rgb will be scaled
    so that the maximum value in the display data will be 1. If
    stretchAll is defined, each color channel will be scaled separately
    so that its maximum value is 1. If bounds is specified, the data will
    be scaled so that lower and upper correspond to 0 and 1, respectively
    . Any values outside of the range (lower, upper) will be clipped.
    '''

    from Numeric import *
    from Spectral.Io.SpyFile import SpyFile
    from exceptions import TypeError

    if not bands:
        bands = []
    monochrome = 0

    if isinstance(source, SpyFile):
        # Figure out which bands to display
        if len(bands) == 0:
            # No bands specified. What should we show?
            if source.metadata.has_key('default bands'):
                try:
                    bands = map(int, source.metadata['default bands'])
                except:
                    pass
            elif source.nBands == 1:
                bands = [0]
        if len(bands) == 0:
            # Pick the first, middle, and last bands
            n = source.nBands
            bands = [0, n / 2, n - 1]
        rgb = source.readBands(bands).astype(Float)
#    elif isinstance(args[0], array):
    else:
        # It should be a Numeric array
        s = source.shape
        if len(s) == 2:
            rgb = source[:, :, NewAxis]
        elif (len(s) == 3 and s[2] == 1):
            rgb = source            
        elif len(s) == 3 and s[2] == 3:
            if len(bands) == 0:
                # keep data as is.
                rgb = source
            elif len(bands) == 3 and bands[0] == 0 and \
                 bands[1] == 1 and bands[2] == 2:
                # Same as first 'if', bands just specified explicitly.
                rgb = source.astype(Float)
            else:
                rgb = take(source, bands, 2).astype(Float)
        else:
            rgb = take(source, [0, s[2] / 2, s[2] - 1], 2).astype(Float)

            

    # If it's either color-indexed or monochrome
    if rgb.shape[2] == 1:
        s = rgb.shape
        if kwargs.has_key("colors"):
            rgb = rgb.astype(Int)
            rgb3 = zeros((s[0], s[1], 3), Int)
            pal = kwargs["colors"]
            for i in range(s[0]):
                for j in range(s[1]):
                    rgb3[i, j] = pal[rgb[i, j, 0]]
            rgb = rgb3
        else:
            monochrome = 1
            rgb = repeat(rgb, 3, 2).astype(Float)
            

    if not kwargs.has_key("colors"):
        # Perform any requested color enhancements.
        if kwargs.has_key("stretch") or not kwargs.has_key("bounds"):
            stretch = 1

        if kwargs.has_key("bounds"):
            # Stretch each color within the value bounds
            (lower, upper) = kwargs["bounds"]
            rgb = (rgb - lower) / (upper - lower)
            rgb = clip(rgb, 0, 1)
        elif kwargs.has_key("stretchAll"):
            # Stretch each color over its full range
            for i in range(rgb.shape[2]):
                mmin = minimum.reduce(ravel(rgb[:, :, i]))
                mmax = maximum.reduce(ravel(rgb[:, :, i]))
                rgb[:, :, i] = (rgb[:, :, i] - mmin) / (mmax - mmin)
        elif stretch or (kwargs.has_key("stretchAll") and monochrome):
            # Stretch so highest color channel value is 1
            mmin = minimum.reduce(ravel(rgb))
            mmax = maximum.reduce(ravel(rgb))
            rgb = (rgb - mmin) / (mmax - mmin)

    return rgb


    def doubleClick(data, pos, *args, **kwargs):
        '''
        Handler for a double-click on the image. The default action is
        to plot the spectrum for the double-clicked pixel.
        '''

        global _xyPlotServer
        _xyPlotServer.plot(data[pos[0], pos[1]], args, kwargs)

        

    
