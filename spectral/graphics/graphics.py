'''
Common functions for extracting and manipulating data for graphical display.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import io
from numbers import Number
import numpy as np
import sys
import time
import warnings

from ..algorithms.spymath import get_histogram_cdf_points
from ..config import spy_colors
from ..image import Image
from ..spectral import settings

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
    from .rasterwindow import RasterWindow
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
    from .hypercube import HypercubeWindow

    if not running_ipython():
        warn_no_ipython()
    check_wx_app()

    window = HypercubeWindow(data, None, -1, *args, **kwargs)
    window.Show()
    window.Raise()
    return window.get_proxy()


def view_nd(data, *args, **kwargs):
    '''
    Creates a 3D window that displays ND data from an image.

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

        `labels` (list):

            List of labels to be displayed next to the axis assigned to a
            feature.  If not specified, the feature index is shown by default.

            The `str()` function will be called on each item of the list so,
            for example, a list of wavelengths can be passed as the labels.

        `size` (2-tuple of ints)

            Specifies the initial size (pixel rows/cols) of the window.

        `title` (string)

            The title to display in the ND window title bar.

    Returns an NDWindowProxy object with a `classes` member to access the
    current class labels associated with data points and a `set_features`
    member to specify which features are displayed.
    '''
    from .ndwindow import NDWindow, validate_args
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

    USAGE: make_pil_image(source [, bands] [stretch=True] [stretch_all=False],
                          [bounds = (lower, upper)] )

    See `get_rgb` for description of arguments.
    '''
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        import Image
        import ImageDraw

    rgb = get_rgb(*args, **kwargs)
    rgb = (rgb * 255).astype(np.ubyte)
    img = Image.fromarray(rgb)
    return img


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


        See :func:`~spectral.graphics.graphics.get_rgb` for descriptions of
        additional keyword arguments.

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
    kwargs = kwargs.copy()
    fmt = kwargs.pop('format', None)

    im = make_pil_image(*(data, bands), **kwargs)
    im.save(filename, fmt, quality=100)


def get_rgb(source, bands=None, **kwargs):
    '''Extract RGB data for display from a SpyFile object or numpy array.

    USAGE: rgb = get_rgb(source [, bands] [, stretch=<arg> | , bounds=<arg>]
                         [, stretch_all=<arg>])

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

        `stretch` (numeric or tuple):

            This keyword specifies two points on the cumulative histogram of
            the input data for performing a linear stretch of RGB value for the
            data. Numeric values given for this parameter are expected to be
            between 0 and 1. This keyword can be expressed in three forms:

            1. As a 2-tuple. In this case the two values specify the lower and
               upper points of the cumulative histogram respectively. The
               specified stretch will be performed independently on each of the
               three color channels unless the `stretch_all` keyword is set to
               True, in which case all three color channels will be stretched
               identically.

            2. As a 3-tuple of 2-tuples. In this case, Each channel will be
               stretched according to its respective 2-tuple in the keyword
               argument.

            3. As a single numeric value. In this case, the value indicates the
               size of the histogram tail to be applied at both ends of the
               histogram for each color channel. `stretch=a` is equivalent to
               `stretch=(a, 1-a)`.

            If neither `stretch` nor `bounds` are specified, then the default
            value of `stretch` defined by `spectral.settings.imshow_stretch`
            will be used.
    
        `bounds` (tuple):

            This keyword functions similarly to the `stretch` keyword, except
            numeric values are in image data units instead of cumulative
            histogram values. The form of this keyword is the same as the first
            two forms for the `stretch` keyword (i.e., either a 2-tuple of
            numbers or a 3-tuple of 2-tuples of numbers).
    
        `stretch_all` (bool):

            If this keyword is True, each color channel will be scaled
            independently.

        `color_scale` (:class:`~spectral.graphics.colorscale.ColorScale`):

            A color scale to be applied to a single-band image.

        `auto_scale` (bool):

            If `color_scale` is provided and `auto_scale` is True, the min/max
            values of the color scale will be mapped to the min/max data
            values.

        `colors` (ndarray):

            If `source` is a single-band integer-valued np.ndarray and this
            keyword is provided, then elements of `source` are assumed to be
            color index values that specify RGB values in `colors`.

    Examples:

    Select color limits corresponding to 2% tails in the data histogram:

    >>> imshow(x, stretch=0.02)

    Same as above but specify upper and lower limits explicitly:

    >>> imshow(x, stretch=(0.02, 0.98))

    Same as above but specify limits for each RGB channel explicitly:

    >>> imshow(x, stretch=((0.02, 0.98), (0.02, 0.98), (0.02, 0.98)))
    '''
    return get_rgb_meta(source, bands, **kwargs)[0]

def _fill_mask(arr, mask, fill_value):
    if mask is None:
        return arr
    arr[mask == 0] = np.array(fill_value) / 255.
    return arr

def get_rgb_meta(source, bands=None, **kwargs):
    '''Same as get_rgb but also returns some metadata.

    Inputs are the same as for get_rgb but the return value is a 2-tuple whose
    first element is the get_rgb return array and whose second element is a
    dictionary containing some metadata values for the data RGB conversion.
    '''
    for k in kwargs:
        if k not in _get_rgb_kwargs:
            raise ValueError('Invalid keyword: {0}'.format(k))

    if bands is None:
        bands = []
    if len(bands) not in (0, 1, 3):
        raise Exception("Invalid number of bands specified.")

    meta = {}
    monochrome = False
    mask = kwargs.get('mask', None)
    bg = kwargs.get('bg', settings.imshow_background_color)

    if isinstance(source, Image) and len(source.shape) == 3:
        # Figure out which bands to display
        s = source.shape
        if len(bands) == 0:
            # No bands specified. What should we show?
            if hasattr(source, 'metadata') and \
              'default bands' in source.metadata:
                try:
                    bands = [int(b) for b in source.metadata['default bands']]
                except:
                    msg = 'Unable to interpret "default bands" in image ' \
                      'metadata. Defaulting to first, middle, & last band.'
                    warnings.warn(msg)
            elif source.shape[-1] == 1:
                bands = [0]
        if len(bands) == 0:
            # Pick the first, middle, and last bands
            n = source.shape[-1]
            bands = [0, n // 2, n - 1]
        rgb = source.read_bands(bands).astype(float)
        meta['bands'] = bands
    else:
        # It should be a numpy array
        if source.ndim == 2:
            source = source[:, :, np.newaxis]
        s = source.shape

        if s[2] == 1:
            if len(bands) == 0:
                bands = [0]
            elif np.max(bands) > 0:
                raise ValueError('Invalid band index for monochrome image.')
        if s[2] == 3 and len(bands) == 0:
            # Keep data as is.
            bands = [0, 1, 2]
        elif s[2] > 3 and len(bands) == 0:
            # More than 3 bands in data but no bands specified so take
            # first, middle, & last bands.
            bands = [0, s[2] / 2, s[2] - 1]

        rgb = np.take(source, bands, 2).astype(float)
        if rgb.ndim == 2:
            rgb = rgb[:, :, np.newaxis]
        meta['bands'] = bands

    color_scale = kwargs.get('color_scale', None)
    auto_scale = kwargs.get('auto_scale', False)

    # If it's either color-indexed or monochrome
    if rgb.shape[2] == 1:
        s = rgb.shape
        if "colors" in kwargs:
            # color-indexed image
            meta['mode'] = 'indexed'
            rgb = rgb.astype(int)
            pal = kwargs["colors"]
            rgb = pal[rgb[:,:,0]] / 255.
            return (_fill_mask(rgb, mask, bg), meta)
        elif color_scale is not None:
            # Colors should be generated from the supplied color scale
            # This section assumes rgb colors in the range 0-255.
            meta['mode'] = 'scaled'
            scale = color_scale
            if auto_scale:
                scale.set_range(min(rgb.ravel()), max(rgb.ravel()))
            rgb3 = np.zeros((s[0], s[1], 3), int)
            rgb3 = np.apply_along_axis(scale, 2, rgb)
            rgb = rgb3.astype(float) / 255.
            return (_fill_mask(rgb, mask, bg), meta)
        else:
            # Only one band of data to display but still need to determine how
            # to scale the data values
            meta['mode'] = 'monochrome'
            monochrome = True
            rgb = np.repeat(rgb, 3, 2).astype(float)

    # Perform any requested color enhancements.

    stretch = kwargs.get('stretch', settings.imshow_stretch)
    stretch_all = kwargs.get('stretch_all', settings.imshow_stretch_all)
    bounds = kwargs.get('bounds', None)

    if  bounds is not None:
        # Data limits for the color stretch are set explicitly
        bounds = np.array(bounds)
        if bounds.shape not in ((2,), (3, 2)):
            msg = '`bounds` keyword must have shape (2,) or (3, 2).'
            raise ValueError(msg)
        if bounds.ndim == 1:
            bounds = np.vstack((bounds,) * 3)
        rgb_lims = bounds
    else:
        # Determine data limits for color stretch from given cumulative
        # histogram values.
        if stretch in (True, False):
            msg = 'Boolean values for `stretch` keyword are deprected. See ' \
              'docstring for `get_rgb`'
            warnings.warn(msg)
            stretch = settings.imshow_stretch
        elif isinstance(stretch, Number):
            if not (0 <= stretch <= 1):
                raise ValueError('Value must be between 0 and 1.')
            stretch = (stretch, 1 - stretch)
        stretch = np.array(stretch)
        if stretch.shape not in ((2,), (3, 2)):
            raise ValueError("`stretch` keyword must be numeric or a " \
                             "sequence with shape (2,) or (3, 2).")
        nondata = kwargs.get('ignore', None)
        if stretch.ndim == 1:
            if monochrome:
                s = get_histogram_cdf_points(rgb[:, :, 0], stretch,
                                             ignore=nondata)
                rgb_lims = [s, s, s]
            elif stretch_all:
                # Stretch each color component independently
                rgb_lims = [get_histogram_cdf_points(rgb[:, :, i], stretch,
                                                     ignore=nondata) \
                            for i in range(3)]
            else:
                # Use a common lower/upper limit for each band by taking
                # the lowest lower limit and greatest upper limit.
                lims = np.array([get_histogram_cdf_points(rgb[:,:,i], stretch,
                                                          ignore=nondata) \
                        for i in range(3)])
                minmax = np.array([lims[:,0].min(), lims[:,1].max()])
                rgb_lims = minmax[np.newaxis, :].repeat(3, axis=0)
        else:
            if monochrome:
                # Not sure why anyone would want separate RGB stretches for
                # a gray-scale image but we'll let them.
                rgb_lims = [get_histogram_cdf_points(rgb[:,:,0], stretch[i],
                                                     ignore=nondata) \
                            for i in range(3)]
            elif stretch_all:
                rgb_lims = [get_histogram_cdf_points(rgb[:,:,i], stretch[i],
                                                     ignore=nondata) \
                            for i in range(3)]
            else:
                msg = 'Can not use common stretch if different stretch ' \
                  ' parameters are given for each color channel.'
                raise ValueError(msg)

    if 'mode' not in meta:
        meta['mode'] = 'rgb'
    meta['rgb range'] = rgb_lims
    for i in range(rgb.shape[2]):
        (lower, upper) = rgb_lims[i]
        span = upper - lower
        if lower == upper:
            rgb[:, :, i] = 0
        else:
            rgb[:, :, i] = np.clip((rgb[:, :, i] - lower) / span, 0, 1)
    return (_fill_mask(rgb, mask, bg), meta)

# For checking if valid keywords were supplied
_get_rgb_kwargs = ('stretch', 'stretch_all', 'bounds', 'colors', 'color_scale',
                   'auto_scale', 'ignore', 'mask', 'bg')

def running_ipython():
    '''Returns True if ipython is running.'''
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def warn_no_ipython():
    '''Warns that user is calling a GUI function outside of ipython.'''
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
    warnings.warn(msg, UserWarning)


def check_wx_app():
    '''Generates a warning if there is not a running wx.App.
    If spectral.START_WX_APP is True and there is no current app, then on will
    be started.
    '''
    import spectral
    import wx
    if wx.GetApp() is None and spectral.settings.START_WX_APP == True:
        warnings.warn('\nThere is no current wx.App object - creating one now.',
                      UserWarning)
        spectral.app = wx.App()

