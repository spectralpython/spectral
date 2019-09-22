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

from __future__ import division, print_function, unicode_literals

import numpy

from warnings import warn


class SpySettings:
    '''Run-time settings for the `spectral` module.

    After importing `spectral`, the settings object is referenced as
    `spectral.settings`.
    
    Noteworthy members:

        `WX_GL_DEPTH_SIZE` (integer, default 24):
    
            Sets the depth (in number of bits) for the OpenGL depth buffer.
            If calls to `view_cube` or `view_nd` result in windows with blank
            canvases, try reducing this value.

        `envi_support_nonlowercase_params` (bool, default False)

            By default, ENVI headers are read with parameter names converted
            to lower case. If this attribute is set to True, parameters will
            be read with original capitalization retained.

        `show_progress` (bool, default True):
    
            Indicates whether long-running algorithms should display progress
            to sys.stdout. It can be useful to set this value to False when
            SPy is embedded in another application (e.g., IPython Notebook).

        `imshow_figure_size` (2-tuple of integers, default `None`):

            Width and height (in inches) of windows opened with `imshow`. If
            this value is `None`, matplotlib's default size is used.
    
        `imshow_background_color` (3-tuple of integers, default (0,0,0)):

            Default color to use for masked pixels in `imshow` displays.

        `imshow_interpolation` (str, default `None`):

            Pixel interpolation to be used in imshow windows. If this value
            is `None`, matplotlib's default interpolation is used. Note that
            zoom windows always use "nearest" interpolation.

        `imshow_stretch`:

            Default RGB linear color stretch to perform.

        `imshow_stretch_all`:

            If True, each color channel limits are determined independently.
    
        `imshow_zoom_figure_width` (int, default `None`):
    
            Width of zoom windows opened from an imshow window. Since zoom
            windows are always square, this is also the window height. If this
            value is `None`, matplotlib's default window size is used.
    
        `imshow_zoom_pixel_width` (int, default 50):

            Number of source image pixel rows and columns to display in a
            zoom window.

        `imshow_float_cmap` (str, default "gray"):

            imshow color map to use with floating point arrays.

        `imshow_class_alpha` (float, default 0.5):

            alpha blending value to use for imshow class overlays

        `imshow_enable_rectangle_selector` (bool, default True):

            Whether to create the rectangle selection tool that enables
            interactive image pixel class labeling. On some OS/backend
            combinations, an exception may be raised when this object is
            created so disabling it allows imshow windows to be created without
            using the selector tool.

        `imshow_disable_mpl_callbacks` (bool, default True):

            If True, several matplotlib keypress event callbacks will be
            disabled to prevent conflicts with callbacks from SPy.  The
            matplotlib callbacks can be set back to their defaults by
            calling `matplotlib.rcdefaults()`.
    '''
    viewer = None
    plotter = None

    # If START_WX_APP is True and there is no current wx.App object when a
    # GUI function is called, then an app object will be created.
    START_WX_APP = True

    # Parameter used by GLCanvas objects in view_cube and view_nd. If the
    # canvas does not render, try reducing this value (e.g., 16).
    WX_GL_DEPTH_SIZE = 24

    envi_support_nonlowercase_params = False

    # Should algorithms show completion progress of algorithms?
    show_progress = True

    # imshow settings
    imshow_figure_size = None
    imshow_background_color = (0, 0, 0)
    imshow_interpolation = None
    imshow_stretch = (0.0, 1.0)
    imshow_stretch_all = True
    imshow_zoom_figure_width = None
    imshow_zoom_pixel_width = 50
    imshow_float_cmap = 'gray'
    imshow_class_alpha = 0.5
    imshow_enable_rectangle_selector = True
    imshow_disable_mpl_callbacks = True
        

settings = SpySettings()

# Default color table
spy_colors = numpy.array([[0, 0, 0],
                          [255, 0, 0],
                          [0, 255, 0],
                          [0, 0, 255],
                          [255, 255, 0],
                          [255, 0, 255],
                          [0, 255, 255],
                          [200, 100, 0],
                          [0, 200, 100],
                          [100, 0, 200],
                          [200, 0, 100],
                          [100, 200, 0],
                          [0, 100, 200],
                          [150, 75, 75],
                          [75, 150, 75],
                          [75, 75, 150],
                          [255, 100, 100],
                          [100, 255, 100],
                          [100, 100, 255],
                          [255, 150, 75],
                          [75, 255, 150],
                          [150, 75, 255],
                          [50, 50, 50],
                          [100, 100, 100],
                          [150, 150, 150],
                          [200, 200, 200],
                          [250, 250, 250],
                          [100, 0, 0],
                          [200, 0, 0],
                          [0, 100, 0],
                          [0, 200, 0],
                          [0, 0, 100],
                          [0, 0, 200],
                          [100, 100, 0],
                          [200, 200, 0],
                          [100, 0, 100],
                          [200, 0, 200],
                          [0, 100, 100],
                          [0, 200, 200]], numpy.int)


def _init():
    '''Basic configuration of the spectral package.'''
    try:
        global settings
        from .graphics import graphics as spygraphics
        from .graphics import spypylab
        settings.plotter = spypylab
        settings.viewer = spygraphics
    except:
        warn('Unable to import or configure pylab plotter.  Spectrum plots '
             'will be unavailable.', UserWarning)

    spectral = __import__(__name__.split('.')[0])
    from .utilities import status
    spectral._status = status.StatusDisplay()


class BandInfo:
    '''A BandInfo object characterizes the spectral bands associated with an
    image. All BandInfo member variables are optional.  For *N* bands, all
    members of type <list> will have length *N* and contain float values.

    =================   =====================================   =======
        Member                  Description                     Default
    =================   =====================================   =======
    centers             List of band centers                    None
    bandwidths          List of band FWHM values                None
    centers_stdevs      List of std devs of band centers        None
    bandwidth_stdevs    List of std devs of bands FWHMs         None
    band_quantity       Image data type (e.g., "reflectance")   ""
    band_unit           Band unit (e.g., "nanometer")           ""
    =================   =====================================   =======
    '''
    def __init__(self):
        self.centers = None
        self.bandwidths = None
        self.centers_stdevs = None
        self.bandwidth_stdevs = None
        self.band_quantity = None
        self.band_unit = None


class Image(object):
    '''spectral.Image is the common base class for spectral image objects.'''

    def __init__(self, params, metadata=None):
        self.bands = BandInfo()
        self.set_params(params, metadata)

    def set_params(self, params, metadata):
        from . import spectral
        import array

        try:
            self.nbands = params.nbands
            self.nrows = params.nrows
            self.ncols = params.ncols
            self.dtype = params.dtype

            if not metadata:
                self.metadata = {}
            else:
                self.metadata = metadata
        except:
            raise

    def params(self):
        '''Return an object containing the SpyFile parameters.'''

        class P:
            pass
        p = P()

        p.nbands = self.nbands
        p.nrows = self.nrows
        p.ncols = self.ncols
        p.metadata = self.metadata
        p.dtype = self.dtype

        return p

    def __repr__(self):
        return self.__str__()


class ImageArray(numpy.ndarray, Image):
    '''ImageArray is an interface to an image loaded entirely into memory.
    ImageArray objects are returned by :meth:`spectral.SpyFile.load`.
    This class inherits from both numpy.ndarray and Image, providing the
    interfaces of both classes.
    '''

    format = 'f'        # Use 4-byte floats for data arrays

    def __new__(subclass, data, spyfile):
        obj = numpy.asarray(data).view(subclass)
        ImageArray.__init__(obj, data, spyfile)
        return obj

    def __init__(self, data, spyfile):
        from .io.spyfile import SpyFile

        # Add param data to Image initializer
        params = spyfile.params()
        params.dtype = data.dtype
        params.swap = 0

        Image.__init__(self, params, spyfile.metadata)
        self.bands = spyfile.bands
        self.filename = spyfile.filename
        self.interleave = 2 # bip

    def __repr__(self):
        lst = numpy.array2string(numpy.asarray(self), prefix="ImageArray(")
        return "{}({}, dtype={})".format('ImageArray', lst, self.dtype.name)

    def __getitem__(self, args):
        # Duplicate the indexing behavior of SpyFile.  If args is iterable
        # with length greater than one, and if not all of the args are
        # scalars, then the scalars need to be replaced with slices.
        import numbers

        try:
            iterator = iter(args)
        except TypeError:
            if isinstance(args, numbers.Number):
                if args == -1:
                    updated_args = slice(args, None)
                else:
                    updated_args = slice(args, args+1)
            else:
                updated_args = args
            return self._parent_getitem(updated_args)

        keep_original_args = True
        updated_args = []
        for arg in iterator:
            if isinstance(arg, numbers.Number):
                if arg == -1:
                    updated_args.append(slice(arg, None))
                else:
                    updated_args.append(slice(arg, arg+1))
            elif isinstance(arg, numpy.bool_):
                updated_args.append(arg)
            else:
                updated_args.append(arg)
                keep_original_args = False

        if keep_original_args:
            updated_args = args
        else:
            updated_args = tuple(updated_args)

        return self._parent_getitem(updated_args)

    def _parent_getitem(self, args):
        return numpy.ndarray.__getitem__(self, args)

    def read_band(self, i):
        '''
        For compatibility with SpyFile objects. Returns arr[:,:,i].squeeze()
        '''
        return numpy.asarray(self[:, :, i].squeeze())

    def read_bands(self, bands):
        '''For SpyFile compatibility. Equivlalent to arr.take(bands, 2)'''
        return numpy.asarray(self.take(bands, 2))

    def read_pixel(self, row, col):
        '''For SpyFile compatibility. Equivlalent to arr[row, col]'''
        return numpy.asarray(self[row, col])

    def read_subregion(self, row_bounds, col_bounds, bands=None):
        '''
        For SpyFile compatibility.

        Equivalent to arr[slice(*row_bounds), slice(*col_bounds), bands],
        selecting all bands if none are specified.
        '''
        if bands:
            return numpy.asarray(self[slice(*row_bounds),
                                      slice(*col_bounds),
                                      bands])
        else:
            return numpy.asarray(self[slice(*row_bounds),
                                      slice(*col_bounds)])

    def read_subimage(self, rows, cols, bands=None):
        '''
        For SpyFile compatibility.

        Equivalent to arr[rows][:, cols][:, :, bands], selecting all bands if
        none are specified.
        '''
        if bands:
            return numpy.asarray(self[rows][:, cols][:, :, bands])
        else:
            return numpy.asarray(self[rows][:, cols])

    def read_datum(self, i, j, k):
        '''For SpyFile compatibility. Equivlalent to arr[i, j, k]'''
        return numpy.asscalar(self[i, j, k])

    def load(self):
        '''For compatibility with SpyFile objects. Returns self'''
        return self

    def asarray(self, writable=False):
        '''Returns an object with a standard numpy array interface.

        The return value is the same as calling `numpy.asarray`, except
        that the array is not writable by default to match the behavior
        of `SpyFile.asarray`.

        This function is for compatibility with SpyFile objects.

        Keyword Arguments:

            `writable` (bool, default False):

                If `writable` is True, modifying values in the returned
                array will result in corresponding modification to the
                ImageArray object.
        '''
        arr = numpy.asarray(self)
        if not writable:
            arr.setflags(write=False)
        return arr

    def info(self):
        s = '\t# Rows:         %6d\n' % (self.nrows)
        s += '\t# Samples:      %6d\n' % (self.ncols)
        s += '\t# Bands:        %6d\n' % (self.shape[2])

        s += '\tData format:  %8s' % self.dtype.name
        return s

    def __array_wrap__(self, out_arr, context=None):
        # The ndarray __array_wrap__ causes ufunc results to be of type
        # ImageArray.  Instead, return a plain ndarray.
        return out_arr

    # Some methods do not call __array_wrap__ and will return an ImageArray.
    # Currently, these need to be overridden individually or with
    # __getattribute__ magic.

    def __getattribute__(self, name):
        if ((name in numpy.ndarray.__dict__) and
            (name not in ImageArray.__dict__)):
            return getattr(numpy.asarray(self), name)

        return super(ImageArray, self).__getattribute__(name)

def open_image(file):
    '''
    Locates & opens the specified hyperspectral image.

    Arguments:

        file (str):
            Name of the file to open.

    Returns:

        SpyFile object to access the file.

    Raises:

        IOError.

    This function attempts to determine the associated file type and open the
    file. If the specified file is not found in the current directory, all
    directories listed in the :const:`SPECTRAL_DATA` environment variable will
    be searched until the file is found.  If the file being opened is an ENVI
    file, the `file` argument should be the name of the header file.
    '''

    import os
    from .io import aviris, envi, erdas, spyfile
    from .io.spyfile import find_file_path

    pathname = find_file_path(file)

    # Try to open it as an ENVI header file.
    try:
        return envi.open(pathname)
    except envi.FileNotAnEnviHeader:
        # It isn't an ENVI file so try another file type
        pass
    except:
        raise

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

    raise IOError('Unable to determine file type or type not supported.')


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
    import numpy as np
    from .io.spyfile import SubImage
    x = (np.array(list(range(nrows + 1))) * float(im.nrows) / nrows).astype(np.int)
    y = (np.array(list(range(ncols + 1))) * float(im.ncols) / ncols).astype(np.int)
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


def load_training_sets(file, image=None):
    '''
    Loads a list of TrainingSet objects from a file.  This function assumes
    that all the sets in the list refer to the same image and mask array.
    If that is not the case, this function should not be used.
    '''
    from .algorithms.algorithms import TrainingClassSet

    ts = TrainingClassSet()
    ts.load(file, image)
    return ts

