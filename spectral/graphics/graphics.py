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

from exceptions import DeprecationWarning
from warnings import warn

_next_window_data_proxy_id = 1
_window_data_proxies = {}

def init_graphics():
    '''Initialize default graphics handlers.'''

    try:
	import spectral
        import pylab
	import spypylab
        pylab.ion()
	spectral.settings.plotter = spypylab
    except:
        print "Unable to initialize Pylab for plotting."
	try:
	    print "Trying Gnuplot..."
	    import spygnuplot
	    spectral.settings.plotter = SpyGnuplot
	    print "Gnuplot initialized."
	except:
	    print "Unable to initialize Gnuplot for plotting."
	    print "No plotters initialized."

    init_wxpython()

def init_wxpython():
    '''Use wxPython for image display.'''
    import spectral
    import spywxpython
    viewer = spywxpython
    viewer.init()
    spectral.settings.viewer = viewer

def initNumTut():
    '''Use NumTut for image display.'''
    import spectral
    import spynumtut
    spectral.settings.viewer = spynumtut

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
	
	    If `stretch_all` evaluates True, the highest value of the data source
	    in each color channel will be set to maximum intensity.
	
	`bounds` (2-tuple of ints):
	
	    Clips the input data at (lower, upper) values.

	`title` (str):
	
	    Text to display in the new window frame.

    `source` is the data source and can be either a :class:`spectral.Image`
    object or a numpy array. If `source` has shape `MxN`, the image will be
    displayed in greyscale. If its shape is `MxNx3`, the three layers/bands will
    be displayed as the red, green, and blue components of the displayed image,
    respectively. If its shape is `MxNxB`, where `B > 3`, the first, middle, and
    last bands will be displayed in the RGB channels, unless `bands` is
    specified.
    '''
    from spectral import settings
    
    # Try to init the graphics thread, if it hasn't already been.
    if not settings.viewer:
	import time
	init_graphics()
	print "Initializing graphics handlers..."
	time.sleep(3)
	try:
	    settings.viewer.view(*args, **kwargs)
	except:
	    print "Error: Failed to display image.  This may be due to the GUI " \
		  "thread taking too long to initialize.  Try calling \"init_graphics()\" " \
		  "to explicitly initialize the GUI thread, then repeat the display command."
    else:
	settings.viewer.view(*args, **kwargs)


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
	    The parameter is a list of 3-tuples defining RGB values, where R, G,
	    and B are in the range [0-255].

	`title` (str):
	
	    Text to display in the new window frame.

    The default color palette used is defined by :obj:`spectral.spy_colors`.
    '''
    from spectral import settings, spy_colors

    if not kwargs.has_key('colors'):
        kwargs['colors'] = spy_colors

    # Try to init the graphics thread, if it hasn't already been.
    if not settings.viewer:
	import time
	init_graphics()
	print "Initializing graphics handlers..."
	time.sleep(3)
	try:
	    settings.viewer.view(*args, **kwargs)
	except:
	    print "Error: Failed to display image.  This may be due to the GUI " \
		  "thread taking too long to initialize.  Try calling \"init_graphics()\" " \
		  "to explicitly initialize the GUI thread, then repeat the display command."
    else:
	settings.viewer.view(*args, **kwargs)
    

def make_pil_image(*args, **kwargs):
    '''
    Save data as a JPEG image file.

    USAGE: view(source [, bands] [stretch = 1] [stretch_all = 1]
                [bounds = (lower, upper)] )

    source is the data source and can be either a SpyFile object or a
    NumPy array.  bands is an optional list which specifies the RGB
    channels to display. If bands is not present and source is a SpyFile
    object, it's metadata dict will be checked if it contains a "default
    bands" item.  Otherwise, the first, middle and last band will be
    displayed. If stretch is defined, the image data will be scaled
    so that the maximum value in the display data will be 1. If
    stretch_all is defined, each color channel will be scaled separately
    so that its maximum value is 1. If bounds is specified, the data will
    be scaled so that lower and upper correspond to 0 and 1, respectively
    . Any values outside of the range (lower, upper) will be clipped.
    '''

    from graphics import get_image_display_data
    import numpy
    from numpy.oldnumeric import transpose
    import StringIO
    import Image, ImageDraw

    rgb = apply(get_image_display_data, args, kwargs)

    if not kwargs.has_key("colors"):
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
    
def save_image(*args, **kwargs):
    '''
    Saves a viewable image to a JPEG (or other format) file.

    Usage::
    
	save_image(data, bands=None, **kwargs)
    
    Arguments:
    
	`data` (:class:`spectral.Image` or :class:`numpy.ndarray`):
	
	    Source image data to display.  `data` can be and instance of a
	    :class:`spectral.Image` (e.g., :class:`spectral.SpyFile` or
	    :class:`spectral.ImageArray`) or a :class:`numpy.ndarray`. `data`
	    must have shape `MxN` or `MxNxB`.  If thes shape is `MxN`, the image
	    will be saved as greyscale (unless keyword `colors` is specified).
	    If the shape is `MxNx3`, it will be interpreted as three `MxN`
	    images defining the R, G, and B channels respectively.  If `B > 3`,
	    the first, middle, and last images in `data` will be used, unless
	    `bands` is specified.
	
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
	
	    If `stretch_all` evaluates True, the highest value of the data source
	    in each color channel will be set to maximum intensity.
	
	`bounds` (2-tuple of ints):
	
	    Clips the input data at (lower, upper) values.

    Examples:
    
	Save a color view of an image by specifying RGB band indices::
	
	    save_image('rgb.jpg', img, [29, 19, 9]])
	
	Save the same image as **png**::
	
	    save_image('rgb.jpg', img, [29, 19, 9]], format='png')
	
	Save classification results using the default color palette (note that
	the color palette must be passed explicitly for `clMap` to be
	interpreted as a color map)::
	
	    save_image('results.jpg', clMap, colors=spectral.spy_colors)

	
	
	
    '''

    im = apply(make_pil_image, args[1:], kwargs)

    if kwargs.has_key("format"):
        fmt = kwargs["format"]
    else:
        fmt = "JPEG"
        
    im.save(args[0], fmt, quality = 100)    

def get_image_display_data(source, bands = None, **kwargs):
    '''
    Extract RGB data to be displayed from a SpyImage or NumPy array.

    USAGE: rgb = get_image_display_data(source [, bands] [stretch = 1]
                    [stretch_all = 1] [bounds = (lower, upper)] )

    source is the data source and can be either a SpyFile object or a
    NumPy array.  bands is an optional list which specifies the RGB
    channels to display. If bands is not present and source is a SpyFile
    object, it's metadata dict will be checked if it contains a "default
    bands" item.  Otherwise, the first, middle and last band will be
    displayed. If stretch is defined, the contents of rgb will be scaled
    so that the maximum value in the display data will be 1. If
    stretch_all is defined, each color channel will be scaled separately
    so that its maximum value is 1. If bounds is specified, the data will
    be scaled so that lower and upper correspond to 0 and 1, respectively
    . Any values outside of the range (lower, upper) will be clipped.
    '''

    from numpy import take, zeros, repeat, ravel, minimum, maximum, clip, \
         float, int, newaxis
    from spectral import Image
    from exceptions import TypeError

    if not bands:
        bands = []
    if len(bands) != 0 and len(bands) != 1 and len(bands) != 3:
        raise Exception("Invalid number of bands specified.")
    monochrome = 0

    if isinstance(source, Image):
        # Figure out which bands to display
        if len(bands) == 0:
            # No bands specified. What should we show?
            if source.metadata.has_key('default bands'):
                try:
                    bands = map(int, source.metadata['default bands'])
                except:
                    pass
            elif source.nbands == 1:
                bands = [0]
        if len(bands) == 0:
            # Pick the first, middle, and last bands
            n = source.nbands
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

    # If it's either color-indexed or monochrome
    if rgb.shape[2] == 1:
        s = rgb.shape
        if kwargs.has_key("colors"):
            rgb = rgb.astype(int)
            rgb3 = zeros((s[0], s[1], 3), int)
            pal = kwargs["colors"]
            for i in range(s[0]):
                for j in range(s[1]):
                    rgb3[i, j] = pal[rgb[i, j, 0]]
            rgb = rgb3
        elif kwargs.has_key("colorScale") and kwargs["colorScale"]:
            # Colors should be generated from the supplied color scale
            # This section assumes rgb colors in the range 0-255.
            rgb = rgb[:,:,0]
            scale = kwargs["colorScale"]
            if kwargs.has_key("autoScale") and kwargs["autoScale"]:
                scale.set_range(min(rgb.ravel()), max(rgb.ravel()))
            rgb3 = zeros((s[0], s[1], 3), int)
            for i in range(s[0]):
                for j in range(s[1]):
                    rgb3[i, j] = scale(rgb[i, j])
            rgb = rgb3.astype(float) / 255.  
        else:
            monochrome = 1
            rgb = repeat(rgb, 3, 2).astype(float)

    if not kwargs.has_key("colors"):
        # Perform any requested color enhancements.
        if kwargs.has_key("stretch") or not kwargs.has_key("bounds"):
            stretch = 1

        if kwargs.has_key("bounds"):
            # Stretch each color within the value bounds
            (lower, upper) = kwargs["bounds"]
            rgb = (rgb - lower) / (upper - lower)
            rgb = clip(rgb, 0, 1)
        elif kwargs.has_key("stretch_all"):
            # Stretch each color over its full range
            for i in range(rgb.shape[2]):
                mmin = minimum.reduce(ravel(rgb[:, :, i]))
                mmax = maximum.reduce(ravel(rgb[:, :, i]))
                rgb[:, :, i] = (rgb[:, :, i] - mmin) / (mmax - mmin)
        elif stretch or (kwargs.has_key("stretch_all") and monochrome):
            # Stretch so highest color channel value is 1
            mmin = minimum.reduce(ravel(rgb))
            mmax = maximum.reduce(ravel(rgb))
            rgb = (rgb - mmin) / (mmax - mmin)

    return rgb

class WindowData(object):
    '''A base class for data rertrievable from open windows.'''
    def __init__(self, proxy_id):
	global _window_data_proxies
	self.window_closed = False
	self.remove_on_close = False
	self._proxy_id = proxy_id
	_window_data_proxies[self._proxy_id] = self

class WindowDataProxy:
    '''A base class for proxying data associated with SPy windows.'''
    def __init__(self):
	self.id = create_window_data_proxy_id()
    def __del__(self):
	global _window_data_proxies
	data = _window_data_proxies.get(self.id)
	if data and data.window_closed == False:
	    data.remove_on_close = True
    def get_data(self):
	global _window_data_proxies
	return _window_data_proxies[self.id]
    def pop(self):
	global _window_data_proxies
	_window_data_proxies.pop(self.id)

def create_window_data_proxy_id():
    '''Returns the ID of a new WindowDataProxy object.'''
    global _next_window_data_proxy_id
    _window_data_proxies[_next_window_data_proxy_id] = None
    _next_window_data_proxy_id += 1
    return _next_window_data_proxy_id - 1

#Deprecated functions

def viewIndexed(*args, **kwargs):
    warn('viewIndexed has been deprecated.  Use view_indexed.',
	 DeprecationWarning)
    return view_indexed(*args, **kwargs)

def saveImage(*args, **kwargs):
    warn('saveImage has been deprecated.  Use save_image.',
	 DeprecationWarning)
    return save_image(*args, **kwargs)

