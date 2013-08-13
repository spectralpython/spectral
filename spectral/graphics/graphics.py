#########################################################################
#
#   graphics.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2001-2010  Thomas Boggs
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
Common functions for extracting and manipulating data for graphical display.
'''

from exceptions import UserWarning
from warnings import warn
import numpy as np
import spectral


class WindowProxy(object):
    '''Base class for proxy objects to access data from display windows.'''
    def __init__(self, window):
        self._window = window


class SpyWindow():
    def get_proxy(self):
        return WindowProxy(self)


def view(*args, **kwargs):
    '''
    Opens a window and displays a raster greyscale or color image.

    Usage::

        view(source, bands=None, **kwargs)

    Arguments:

        `source` (:class:`spectral.Image` or :class:`numpy.ndarray`):

            Source image data to display.  `source` can be and instance of a
            :class:`spectral.Image` (e.g., :class:`spectral.SpyFile` or
            :class:`spectral.ImageArray`) or a :class:`numpy.ndarray`. `source`
            must have shape `MxN` or `MxNxB`.

        `bands` (3-tuple of ints):

            Optional list of indices for bands to display in the red, green,
            and blue channels, respectively.

    Keyword Arguments:

        `stretch` (bool):

            If `stretch` evaluates True, the highest value in the data source
            will be scaled to maximum color channel intensity.

        `stretch_all` (bool):

            If `stretch_all` evaluates True, the highest value of the data
            source in each color channel will be set to maximum intensity.

        `bounds` (2-tuple of ints):

            Clips the input data at (lower, upper) values.

        `title` (str):

            Text to display in the new window frame.

    `source` is the data source and can be either a :class:`spectral.Image`
    object or a numpy array. If `source` has shape `MxN`, the image will be
    displayed in greyscale. If its shape is `MxNx3`, the three layers/bands
    will be displayed as the red, green, and blue components of the displayed
    image, respectively. If its shape is `MxNxB`, where `B > 3`, the first,
    middle, and last bands will be displayed in the RGB channels, unless
    `bands` is specified.
    '''
    import graphics
    from spectral.spectral import Image
    from spectral.graphics.rasterwindow import RasterWindow

    if not running_ipython():
        warn_no_ipython()
    check_wx_app()

    rgb = get_rgb(*args, **kwargs)

    # To plot pixel spectrum on double-click, create a reference
    # back to the original SpyFile object.
    if isinstance(args[0], Image):
        kwargs["data source"] = args[0]

    if "colors" not in kwargs:
        rgb = (rgb * 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)

    frame = RasterWindow(None, -1, rgb, **kwargs)
    frame.Raise()
    frame.Show()
    return frame.get_proxy()


def view_cube(data, *args, **kwargs):
    '''Renders an interactive 3D hypercube in a new window.

    Arguments:

        `data` (:class:`spectral.Image` or :class:`numpy.ndarray`):

            Source image data to display.  `data` can be and instance of a
            :class:`spectral.Image` (e.g., :class:`spectral.SpyFile` or
            :class:`spectral.ImageArray`) or a :class:`numpy.ndarray`. `source`
            must have shape `MxN` or `MxNxB`.

    Keyword Arguments:

        `bands` (3-tuple of ints):

            3-tuple specifying which bands from the image data should be
            displayed on top of the cube.

        `top` (:class:`numpy.ndarray` or :class:`PIL.Image`):

            Data to display on top of the cube. This will supercede the
            `bands` keyword.

        `scale` (:class:`spectral.ColorScale`)

            A color scale to be used for color in the sides of the cube. If
            this keyword is not specified,
            :obj:`spectral.graphics.colorscale.defaultColorScale` is used.

        `size` (2-tuple of ints):

            Width and height (in pixels) for initial size of the new window.

        `background` (3-tuple of floats):

            Background RGB color of the scene. Each value should be in the
            range [0, 1]. If not specified, the background will be black.

        `title` (str):

            Title text to display in the new window frame.

    This function opens a new window, renders a 3D hypercube, and accepts
    keyboard input to manipulate the view of the hypercube.  Accepted keyboard
    inputs are printed to the console output.  Focus must be on the 3D window
    to accept keyboard input.
    '''
    from spectral.graphics.hypercube import HypercubeWindow

    if not running_ipython():
        warn_no_ipython()
    check_wx_app()

    window = HypercubeWindow(data, None, -1, *args, **kwargs)
    window.Show()
    window.Raise()
    return window.get_proxy()


def view_nd(data, *args, **kwargs):
    '''Creates a 3D window that displays ND data from an image.

    Arguments:

        `data` (:class:`spectral.ImageArray` or :class:`numpy.ndarray`):

            Source image data to display.  `data` can be and instance of a
            :class:`spectral.ImageArray or a :class:`numpy.ndarray`. `source`
            must have shape `MxNxB`, where M >= 3.

    Keyword Arguments:

        `classes` (:class:`numpy.ndarray`):

            2-dimensional array of integers specifying the classes of each
            pixel in `data`. `classes` must have the same dimensions as the
            first two dimensions of `data`.

        `features` (list or list of integer lists):

            This keyword specifies which bands/features from `data` should be
            displayed in the 3D window. It must be defined as one of the
            following:

            #. A length-3 list of integer feature IDs. In this case, the data
               points will be displayed in the positive x,y,z octant using
               features associated with the 3 integers.

            #. A length-6 list of integer feature IDs. In this case, each
               integer specifies a single feature index to be associated with
               the coordinate semi-axes x, y, z, -x, -y, and -z (in that
               order). Each octant will display data points using the features
               associated with the 3 semi-axes for that octant.

            #. A length-8 list of length-3 lists of integers. In this case,
               each length-3 list specfies the features to be displayed in a
               single octants (the same semi-axis can be associated with
               different features in different octants).  Octants are ordered
               starting with the postive x,y,z octant and procede
               counterclockwise around the z-axis, then procede similarly
               around the negative half of the z-axis.  An octant triplet can
               be specified as None instead of a list, in which case nothing
               will be rendered in that octant.

        `size` (2-tuple of ints)

            Specifies the initial size (pixel rows/cols) of the window.

        `title` (string)

            The title to display in the ND window title bar.

    Returns an NDWindowProxy object with a `classes` member to access the
    current class labels associated with data points and a `set_features`
    member to specify which features are displayed.
    '''
    import spectral
    import time
    from spectral.graphics.ndwindow import NDWindow, validate_args

    if not running_ipython():
        warn_no_ipython()
    check_wx_app()

    validate_args(data, *args, **kwargs)
    window = NDWindow(data, None, -1, *args, **kwargs)
    window.Show()
    window.Raise()
    return window.get_proxy()


def view_indexed(*args, **kwargs):
    '''
    Opens a window and displays a raster image for the provided color map data.

    Usage::

        view_indexed(data, **kwargs)

    Arguments:

        `data` (:class:`numpy.ndarray`):

            An `MxN` array of integer values that correspond to colors in a
            color palette.

    Keyword Arguments:

        `colors` (list of 3-tuples of ints):

            This parameter provides an alternate color map to use for display.
            The parameter is a list of 3-tuples defining RGB values, where R,
            G, and B are in the range [0-255].

        `title` (str):

            Text to display in the new window frame.

    The default color palette used is defined by :obj:`spectral.spy_colors`.
    '''
    from spectral import settings, spy_colors

    if not running_ipython():
        warn_no_ipython()
    check_wx_app()

    if 'colors' not in kwargs:
        kwargs['colors'] = spy_colors

    return view(*args, **kwargs)

def imshow(data, bands=None, **kwargs):
    '''A wrapper around matplotlib's imshow for multi-band images.

    Arguments:

        `data` (SpyFile or ndarray):

            Can have shape (R, C) or (R, C, B).

        `bands` (tuple of integers, optional)

            If `bands` has 3 values, the bands specified are extracted from
            `data` to be plotted as the red, green, and blue colors,
            respectively. If it contains a single value, then a single band
            will be extracted from the image.

    Keyword Arguments:

        show_xaxis (bool, default True):

            Indicates whether to display x-axis ticks and labels.

        show_yaxis (bool, default True):

            Indicates whether to display y-axis ticks and labels.

    This function is a wrapper around
    :func:`~spectral.graphics.graphics.get_rgb` and matplotlib's imshow.
    All keyword arguments other than those described above are passed on to
    the wrapped functions.

    This function defaults the color scale (imshow's "cmap" keyword) to
    "gray". To use imshow's default color scale, call this function with
    keyword `cmap=None`.
    '''
    import matplotlib.pyplot as plt

    show_xaxis = True
    show_yaxis = True
    if 'show_xaxis' in kwargs:
        show_xaxis = kwargs.pop('show_xaxis')
    if 'show_yaxis' in kwargs:
        show_yaxis = kwargs.pop('show_yaxis')

    rgb_kwargs = {}
    for k in ['stretch', 'stretch_all', 'bounds']:
        if k in kwargs:
            rgb_kwargs[k] = kwargs.pop(k)
    
    imshow_kwargs = {'cmap': 'gray'}
    imshow_kwargs.update(kwargs)

    rgb = get_rgb(data, bands, **rgb_kwargs)

    # Allow matplotlib.imshow to apply a color scale to single-band image.
    if len(data.shape) == 2:
        rgb = rgb[:, :, 0]

    ax = plt.imshow(rgb, **imshow_kwargs)
    if show_xaxis == False:
        plt.gca().xaxis.set_visible(False)
    if show_yaxis == False:
        plt.gca().yaxis.set_visible(False)
    return ax

def make_pil_image(*args, **kwargs):
    '''Creates a PIL Image object.

    USAGE: make_pil_image(source [, bands] [stretch = 1] [stretch_all = 1]
                          [bounds = (lower, upper)] )

    See `get_rgb` for description of arguments.
    '''

    import numpy
    from numpy.oldnumeric import transpose
    import StringIO
    import Image
    import ImageDraw

    rgb = get_rgb(*args, **kwargs)

    if "colors" not in kwargs:
        rgb = (rgb * 255).astype(numpy.ubyte)
    else:
        rgb = rgb.astype(numpy.ubyte)
    rgb = transpose(rgb, (1, 0, 2))
    im = Image.new("RGB", rgb.shape[:2])
    draw = ImageDraw.ImageDraw(im)

    # TO DO:
    # Find a more efficient way to write data to the PIL image below.
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            draw.point((i, j), tuple(rgb[i, j]))

    return im


def save_rgb(filename, data, bands=None, **kwargs):
    '''
    Saves a viewable image to a JPEG (or other format) file.

    Usage::

        save_rgb(filename, data, bands=None, **kwargs)

    Arguments:

        `filename` (str):

            Name of image file to save (e.g. "rgb.jpg")

        `data` (:class:`spectral.Image` or :class:`numpy.ndarray`):

            Source image data to display.  `data` can be and instance of a
            :class:`spectral.Image` (e.g., :class:`spectral.SpyFile` or
            :class:`spectral.ImageArray`) or a :class:`numpy.ndarray`. `data`
            must have shape `MxN` or `MxNxB`.  If thes shape is `MxN`, the
            image will be saved as greyscale (unless keyword `colors` is
            specified). If the shape is `MxNx3`, it will be interpreted as
            three `MxN` images defining the R, G, and B channels respectively.
            If `B > 3`, the first, middle, and last images in `data` will be
            used, unless `bands` is specified.

        `bands` (3-tuple of ints):

            Optional list of indices for bands to use in the red, green,
            and blue channels, respectively.

    Keyword Arguments:

        `format` (str):

            The image file format to create.  Must be a format recognized by
            :mod:`PIL` (e.g., 'png', 'tiff', 'bmp').  If `format` is not
            provided, 'jpg' is assumed.

        `colors` (list of 3-tuples of ints):

            If this keyword is provided, `data` is interpeted to be a color map
            into the `colors` color palette. This is the same `colors` keyword
            used by the :func:`spectral.view_indexed` function. The parameter
            is a list of 3-tuples defining RGB values, where R, G, and B are
            in the range [0-255].

        `stretch` (bool):

            If `stretch` evaluates True, the highest value in the data source
            will be scaled to maximum color channel intensity.

        `stretch_all` (bool):

            If `stretch_all` evaluates True, the highest value of the data
            source in each color channel will be set to maximum intensity.

        `bounds` (2-tuple of ints):

            Clips the input data at (lower, upper) values.

    Examples:

        Save a color view of an image by specifying RGB band indices::

            save_image('rgb.jpg', img, [29, 19, 9]])

        Save the same image as **png**::

            save_image('rgb.png', img, [29, 19, 9]], format='png')

        Save classification results using the default color palette (note that
        the color palette must be passed explicitly for `clMap` to be
        interpreted as a color map)::

            save_image('results.jpg', clMap, colors=spectral.spy_colors)
    '''
    im = make_pil_image(*(data, bands), **kwargs)

    if "format" in kwargs:
        im.save(filename, kwargs['format'], quality=100)
    else:
        im.save(filename)


def get_rgb(source, bands=None, **kwargs):
    '''Extract RGB data for display from a SpyFile object or numpy array.

    USAGE: rgb = get_rgb(source [, bands] [stretch = 1]
                         [stretch_all = 1] [bounds = (lower, upper)] )

    Arguments:

        `source` (:class:`spectral.SpyFile` or :class:`numpy.ndarray`):

            Data source from which to extract the RGB data.

        `bands` (list of `int`) (optional):

            Optional triplet of indices which specifies the bands to extract
            for the red, green, and blue components, respectively. If this
            arg is not given, SpyFile object, it's metadata dict will be
            checked to see if it contains a "default bands" item.  If it does
            not, then first, middle and last band will be returned.

    Keyword Arguments:

        `stretch` (bool, default True):

            If the `stretch` keyword is True, the RGB values will be scaled
            so the maximum value in the returned array will be 1.

        `stretch_all` (bool, default False):

            If this keyword is True, each color channel will be scaled
            separately such that its maximum value is 1.

        `bounds` (2-tuple of scalars):

            If `bounds` is specified, the data will be scaled so that `lower`
            and `upper` correspond to 0 and 1, respectively. Any values outside
            of the range (`lower`, `upper`) will be clipped.
    '''

    from numpy import (take, zeros, repeat, ravel, minimum, maximum, clip,
                       float, int, newaxis)
    from spectral.spectral import Image
    from exceptions import TypeError

    if not bands:
        bands = []
    if len(bands) != 0 and len(bands) != 1 and len(bands) != 3:
        raise Exception("Invalid number of bands specified.")
    monochrome = 0

    if isinstance(source, Image) and len(source.shape) == 3:
        # Figure out which bands to display
        if len(bands) == 0:
            # No bands specified. What should we show?
            if hasattr(source, 'metadata') and \
              'default bands' in source.metadata:
                try:
                    bands = [int(b) for b in source.metadata['default bands']]
                except:
                    pass
            elif source.shape[-1] == 1:
                bands = [0]
        if len(bands) == 0:
            # Pick the first, middle, and last bands
            n = source.shape[-1]
            bands = [0, n / 2, n - 1]
        rgb = source.read_bands(bands).astype(float)
    else:
        # It should be a numpy array
        s = source.shape
        if len(s) == 2:
            rgb = source[:, :, newaxis]
        elif (len(s) == 3 and s[2] == 1):
            rgb = source
        elif len(s) == 3:
            if s[2] == 3:
                if len(bands) == 0:
                    # keep data as is.
                    rgb = source.astype(float)
                elif len(bands) == 3:
                    if bands[0] == 0 and bands[1] == 1 and bands[2] == 2:
                        # Same as first 'if', bands just explicit.
                        rgb = source.astype(float)
                    else:
                        rgb = take(source, bands, 2).astype(float)
            elif s[2] > 3 and (len(bands) == 1 or len(bands) == 3):
                rgb = take(source, bands, 2).astype(float)
            else:
                rgb = take(source, [0, s[2] / 2, s[2] - 1], 2).astype(float)
        else:
            raise Exception('Invalid array shape for image display')

    if 'colorScale' in kwargs:
        color_scale = kwargs['colorScale']
        warn('Keyword "colorScale" is deprecated. Use "color_scale"',
             UserWarning)
    else:
        color_scale = kwargs.get('color_scale', None)

    if 'autoScale' in kwargs:
        auto_scale = kwargs['autoScale']
        warn('Keyword "autoScale" is deprecated. Use "auto_scale"',
             UserWarning)
    else:
        auto_scale = kwargs.get('auto_scale', False)

    # If it's either color-indexed or monochrome
    if rgb.shape[2] == 1:
        s = rgb.shape
        if "colors" in kwargs:
            rgb = rgb.astype(int)
            rgb3 = zeros((s[0], s[1], 3), int)
            pal = kwargs["colors"]
            for i in range(s[0]):
                for j in range(s[1]):
                    rgb3[i, j] = pal[rgb[i, j, 0]]
            rgb = rgb3
        elif color_scale is not None:
            # Colors should be generated from the supplied color scale
            # This section assumes rgb colors in the range 0-255.
            rgb = rgb[:, :, 0]
            scale = color_scale
            if auto_scale:
                scale.set_range(min(rgb.ravel()), max(rgb.ravel()))
            rgb3 = zeros((s[0], s[1], 3), int)
            for i in range(s[0]):
                for j in range(s[1]):
                    rgb3[i, j] = scale(rgb[i, j])
            rgb = rgb3.astype(float) / 255.
        else:
            monochrome = 1
            rgb = repeat(rgb, 3, 2).astype(float)

    if "colors" not in kwargs:
        # Perform any requested color enhancements.
        if "stretch" in kwargs or "bounds" not in kwargs:
            stretch = 1

        if "bounds" in kwargs:
            # Stretch each color within the value bounds
            (lower, upper) = kwargs["bounds"]
            rgb = (rgb - lower) / (upper - lower)
            rgb = clip(rgb, 0, 1)
        elif "stretch_all" in kwargs:
            # Stretch each color over its full range
            for i in range(rgb.shape[2]):
                mmin = minimum.reduce(ravel(rgb[:, :, i]))
                mmax = maximum.reduce(ravel(rgb[:, :, i]))
                rgb[:, :, i] = (rgb[:, :, i] - mmin) / (mmax - mmin)
        elif stretch or ("stretch_all" in kwargs and monochrome):
            # Stretch so highest color channel value is 1
            mmin = minimum.reduce(ravel(rgb))
            mmax = maximum.reduce(ravel(rgb))
            rgb = (rgb - mmin) / (mmax - mmin)

    return rgb


def running_ipython():
    '''Returns True if ipython is running.'''
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def warn_no_ipython():
    '''Warns that user is calling a GUI function outside of ipython.'''
    import sys
    msg = '''
#############################################################################
SPy graphics functions are inteded to be run from IPython with the
`pylab` mode set for wxWindows.  For example,

    # ipython --pylab=WX

GUI functions will likely not function properly if you aren't running IPython
or haven't started it configured for pylab and wx.
#############################################################################
'''

    if sys.platform == 'darwin':
        msg += '''
NOTE: If you are running on Mac OS X and receive an error message
stating the following:

    "PyNoAppError: The wx.App object must be created first!",

You can avoid this error by running the following commandes immediately after
starting your ipython session:

    In [1]: import wx

    In [2]: app = wx.App()
#############################################################################
'''
    warn(msg, UserWarning)


def check_wx_app():
    '''Generates a warning if there is not a running wx.App.
    If spectral.START_WX_APP is True and there is no current app, then on will
    be started.
    '''
    import spectral
    import wx
    if wx.GetApp() is None and spectral.settings.START_WX_APP == True:
        warn('\nThere is no current wx.App object - creating one now.',
             UserWarning)
        spectral.app = wx.App()

#Deprecated functions


def hypercube(*args, **kwargs):
    warn('Function `hypercube` has been deprecated.  Use `view_cube`.',
         UserWarning)
    return view_cube(*args, **kwargs)


def ndwindow(*args, **kwargs):
    warn('Function `ndwindow` has been deprecated.  Use `view_nd`.',
         UserWarning)
    return view_nd(*args, **kwargs)

def save_image(*args, **kwargs):
    '''See function `save_rgb`.'''
    msg = 'Function `save_image` has been deprecated.  It has been' \
         ' replaced by `save_rgb`.'
    warn(msg, UserWarning)
    return save_rgb(*args, **kwargs)
    
def get_image_display_data(source, bands=None, **kwargs):
    '''Deprecated function. Use `get_rgb` instead.'''
    msg = 'Function `get_image_display_data` has been deprecated.  It has' \
          ' been replaced by `get_rgb`.'
    warn(msg, UserWarning)
    return get_rgb(source, bands, **kwargs)
