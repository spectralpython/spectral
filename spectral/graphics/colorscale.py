'''
Code for converting pixel data to RGB values.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


class ColorScale:
    '''
    A color scale class to map scalar values to rgb colors.  The class allows
    associating colors with particular scalar values, setting a background
    color (for values below threshold), andadjusting the scale limits. The
    :meth:`__call__` operator takes a scalar input and returns the
    corresponding color, interpolating between defined colors.
    '''
    def __init__(self, levels, colors, num_tics=0):
        '''
        Creates the ColorScale.

        Arguments:

            `levels` (list of numbers):

                Scalar levels to which the `colors` argument will correspond.

            `colors` (list of 3-tuples):

                RGB 3-tuples that define the colors corresponding to `levels`.

            `num_tics` (int):

                The total number of colors in the scale, not including the
                background color.  This includes the colors given in the
                `colors` argument, as well as interpolated color values. If
                not specified, only the colors in the `colors` argument will
                be used (i.e., num_tics = len(colors).
        '''
        import numpy as np
        if len(colors.shape) != 2 or colors.shape[1] != 3:
            raise 'colors array has invalid shape.'
        if len(levels) != colors.shape[0]:
            raise 'Number of scale levels and colors do not match.'

        if num_tics == 0:
            num_tics = len(colors)
        if num_tics < 2:
            msg = 'There must be at least two tics in the color scale.'
            raise ValueError(msg)

        # Make sure scale levels are floats
        if type(levels) in (list, tuple):
            levels = [float(x) for x in levels]
        elif isinstance(levels, np.ndarray):
            levels = levels.astype(float)

        self.span = levels[-1] - levels[0]
        self.max = levels[-1]
        self.min = levels[0]
        self.tics = np.linspace(self.min, self.max, num_tics)
        self.colorTics = np.zeros((len(self.tics), 3), int)
        self.size = len(self.tics)
        self.bgColor = np.array([0, 0, 0])

        j = 1
        dcolor = colors[1] - colors[0]
        dlevel = levels[1] - levels[0]
        for i in range(len(self.tics)):
            while self.tics[i] >= levels[j] and j < len(levels) - 1:
                j += 1
                dcolor = colors[j] - colors[j - 1]
                dlevel = levels[j] - levels[j - 1]
            self.colorTics[i] = (colors[j - 1] + (self.tics[i] - levels[j - 1])
                                 / dlevel * dcolor).astype(int)

    def __call__(self, val):
        '''Returns the scale color associated with the given value.'''
        if val < self.min:
            return self.bgColor
        elif val >= self.max:
            return self.colorTics[-1]
        else:
            return self.colorTics[int((float(val) - self.min)
                                  / self.span * self.size)]

    def set_background_color(self, color):
        '''Sets RGB color used for values below the scale minimum.

        Arguments:

            `color` (3-tuple): An RGB triplet
        '''
        if type(color) in (list, tuple):
            color = np.array(color)
        if len(color.shape) != 1 or color.shape[0] != 3:
            raise 'Color value must be have exactly 3 elements.'
        self.bgColor = color

    def set_range(self, min, max):
        '''Sets the min and max values of the color scale.

        The distribution of colors within the scale will stretch or shrink
        accordingly.
        '''
        self.min = min
        self.max = max
        self.span = max - min


def create_default_color_scale(ntics=0):
    '''Returns a black-blue-green-red-yellow-white color scale.

    Arguments:

            `ntics` (integer):

                Total number of colors in the scale. If this value is 0, no
                interpolated colors will be used.
    '''
    mycolors = np.array([[0, 0, 0],
                         [0, 0, 255],
                         [0, 255, 0],
                         [255, 0, 0],
                         [255, 255, 0],
                         [255, 255, 255]])

    if ntics != 0 and ntics < len(mycolors):
        raise ValueError('Any non-zero value of `ntics` must be greater than'
                         ' {}.'.format(len(mycolors)))
    levels = np.array([0., 10., 20., 30., 40., 50.])
    scale = ColorScale(levels, mycolors, ntics)
    return scale


default_color_scale = create_default_color_scale()
