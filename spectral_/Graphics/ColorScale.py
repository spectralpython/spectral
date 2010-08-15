#########################################################################
#
#   ColorScale.py - This file is part of the Spectral Python (SPy) package.
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

class ColorScale:
    '''
    A color scale class to map scalar values to rgb colors.  The class
    allows associating colors with particular scalar values, setting a
    background color (for values below threshold), adjusting the scale
    limits.  The class\' __call__ operator takes scalar inputs and
    returns the associated color, interpolating between defined colors.
    '''    
    def __init__(self, levels, colors, numTics = 256):
        '''
        Creates the ColorScale.

        USAGE: scale = ColorScale(levels, colors [numTics = 256])

        ARGUMENTS:
            levels          An array of scalar levels to which the colors
                            argument will correspond.
            colors          An array of rgb 3-tuples that define the
                            colors corresponding to levels.
            numTicks        The total number of colors in the scale, not
                            including the background color.  This
                            includes the colors given in the arguement,
                            as well as interpolated color values.
        '''        
        from numpy.oldnumeric import array, zeros, Float, Int, ArrayType
        if len(colors.shape) != 2 or colors.shape[1] != 3:
            raise 'colors array has invalid shape.'
        if len(levels) != colors.shape[0]:
            raise 'Number of scale levels and colors do not match.'

        # Make sure scale levels are floats
        if type(levels) == list:
            levels = [float(x) for x in levels]
        elif isinstance(levels, ArrayType):
            levels = levels.astype(Float)
            
        self.span = levels[-1] - levels[0]
        self.max = levels[-1]
        self.min = levels[0]
        self.tics = array(range(numTics), Float) * (self.span / numTics)
        self.colorTics = zeros([self.tics.shape[0], 3], Int)
        self.size = len(self.tics)
        self.bgColor = array([0, 0, 0])
        
        j = 1
        dcolor = colors[1] - colors[0]
        dlevel = levels[1] - levels[0]
        for i in range(len(self.tics)):
            while self.tics[i] >= levels[j] and j < len(levels) - 1:
                j += 1
                dcolor = colors[j] - colors[j - 1]
                dlevel = levels[j] - levels[j - 1]
            self.colorTics[i] = (colors[j - 1] + (self.tics[i] - levels[j - 1]) \
                                                 / dlevel * dcolor).astype(Int)

    def __call__(self, val):
        '''
        Return the scale color associated with the given value.
        '''
        if val < self.min:
            return self.bgColor
        elif val >= self.max:
            return self.colorTics[-1]
        else:
            return self.colorTics[int((float(val) - self.min) / self.span * self.size)]

    def setBackgroundColor(c):
        '''
        Sets rgb color used for values below the scale minimum.
        '''
        if type(c) == list:
            c = array(c)
        if len(c.shape) != 1 or c.shape[0] != 3:
            raise 'Color value must be have exactly 3 elements.'
        self.bgColor = c

    def setRange(self, min, max):
        '''
        Set the min mand max values of the color scale.  The distribution
        of colors within the scale will stretch or shrink accordingly.
        '''
        self.min = min
        self.max = max
        self.span = max - min


def createDefaultColorScale():
    '''Returns a black-blue-green-red-white color scale.'''
    from numpy.oldnumeric import array
    mycolors = array([[  0,   0,   0],
                      [  0,   0, 255],
                      [  0, 255,   0],
                      [255,   0,   0],
                      [255, 255,   0],
                      [255, 255, 255]])
    levels = array([0., 10., 20., 30., 40., 50.])
    scale = ColorScale(levels, mycolors)
    return scale

defaultColorScale = createDefaultColorScale()
