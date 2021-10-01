'''
:class:`~spectral.SpyFile` is the base class for creating objects to read
hyperspectral data files.  When a :class:`~spectral.SpyFile` object is created,
it provides an interface to read data from a corresponding file.  When an image
is opened, the actual object returned will be a subclass of
:class:`~spectral.SpyFile` (BipFile, BilFile, or BsqFile) corresponding to the
interleave of the data within the image file.

Let's open our sample image.

.. ipython::

    In [1]: from spectral import *

    In [2]: img = open_image('92AV3C.lan')

    In [3]: img.__class__
    Out[3]: spectral.io.bilfile.BilFile

    In [4]: print(img)
            Data Source:   '/Users/thomas/spectral_data/92AV3C.lan'
            # Rows:            145
            # Samples:         145
            # Bands:           220
            Interleave:        BIL
            Quantization:  16 bits
            Data format:     int16

The image was not located in the working directory but it was still opened
because it was in a directory specified by the *SPECTRAL_DATA* environment
variable.  Because the image pixel data are interleaved by line, the *image*
function returned a *BilFile* instance.

Since hyperspectral image files can be quite large, only
metadata are read from the file when the :class:`~spectral.SpyFile` object is
first created. Image data values are only read when specifically requested via
:class:`~spectral.SpyFile` methods.  The :class:`~spectral.SpyFile` class
provides a subscript operator that behaves much like the numpy array subscript
operator. The :class:`~spectral.SpyFile` object is subscripted as an *MxNxB*
array where *M* is the number of rows in the image, *N* is the number of
columns, and *B* is thenumber of bands.

.. ipython::

    In [5]: img.shape
    Out[5]: (145, 145, 220)

    In [6]: pixel = img[50,100]

    In [7]: pixel.shape
    Out[7]: (220,)

    In [8]: band6 = img[:,:,5]

    In [9]: band6.shape
    Out[9]: (145, 145, 1)

The image data values were not read from the file until the subscript operator
calls were performed.  Note that since Python indices start at 0,
``img[50,100]`` refers to the pixel at 51st row and 101st column of the image.
Similarly, ``img[:,:,5]`` refers to all the rows and columns for the 6th band
of the image.

:class:`~spectral.SpyFile` subclass instances returned for particular image
files will also provide the following methods:

==============   ===============================================================
   Method                               Description
==============   ===============================================================
read_band        Reads a single band into an *MxN* array
read_bands       Reads multiple bands into an *MxNxC* array
read_pixel       Reads a single pixel into a length *B* array
read_subregion   Reads multiple bands from a rectangular sub-region of the image
read_subimage    Reads specified rows, columns, and bands
==============   ===============================================================

:class:`~spectral.SpyFile` objects have a ``bands`` member, which is an
instance of a :class:`~spectral.BandInfo` object that contains optional
information about the images spectral bands.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import array
import numpy as np
import os
import warnings

import spectral as spy
from .. import SpyException
from ..image import Image, ImageArray
from ..utilities.errors import has_nan, NaNValueWarning
from ..utilities.python23 import typecode, tobytes, frombytes


class FileNotFoundError(SpyException):
    pass

class InvalidFileError(SpyException):
    '''Raised when file contents are invalid for the exepected file type.'''
    pass

def find_file_path(filename):
    '''
    Search cwd and SPECTRAL_DATA directories for the given file.
    '''
    pathname = None
    dirs = [os.curdir]
    if 'SPECTRAL_DATA' in os.environ:
        dirs += os.environ['SPECTRAL_DATA'].split(os.pathsep)
    for d in dirs:
        testpath = os.path.join(d, filename)
        if os.path.isfile(testpath):
            pathname = testpath
            break
    if not pathname:
        msg = 'Unable to locate file "%s". If the file exists, ' \
          'use its full path or place its directory in the ' \
          'SPECTRAL_DATA environment variable.'  % filename
        raise FileNotFoundError(msg)
    return pathname


class SpyFile(Image):
    '''A base class for accessing spectral image files'''

    def __init__(self, params, metadata=None):
        Image.__init__(self, params, metadata)
        # Number by which to divide values read from file.
        self.scale_factor = 1.0

    def set_params(self, params, metadata):
        Image.set_params(self, params, metadata)

        try:
            self.filename = params.filename
            self.offset = params.offset
            self.byte_order = params.byte_order
            if spy.byte_order != self.byte_order:
                self.swap = 1
            else:
                self.swap = 0
            self.sample_size = np.dtype(params.dtype).itemsize

            self.fid = open(find_file_path(self.filename), "rb")

            # So that we can use this more like a Numeric array
            self.shape = (self.nrows, self.ncols, self.nbands)

        except:
            raise

    def transform(self, xform):
        '''Returns a SpyFile image with the linear transform applied.'''
        # This allows a LinearTransform object to take the SpyFile as an arg.
        return transform_image(xform, self)

    def __str__(self):
        '''Prints basic parameters of the associated file.'''
        s = '\tData Source:   \'%s\'\n' % self.filename
        s += '\t# Rows:         %6d\n' % (self.nrows)
        s += '\t# Samples:      %6d\n' % (self.ncols)
        s += '\t# Bands:        %6d\n' % (self.shape[2])
        if self.interleave == spy.BIL:
            interleave = 'BIL'
        elif self.interleave == spy.BIP:
            interleave = 'BIP'
        else:
            interleave = 'BSQ'
        s += '\tInterleave:     %6s\n' % (interleave)
        s += '\tQuantization: %3d bits\n' % (self.sample_size * 8)

        s += '\tData format:  %8s' % np.dtype(self.dtype).name
        return s

    def load(self, **kwargs):
        '''Loads entire image into memory in a :class:`spectral.image.ImageArray`.

        Keyword Arguments:

            `dtype` (numpy.dtype):

                An optional dtype to which the loaded array should be cast.

            `scale` (bool, default True):

                Specifies whether any applicable scale factor should be applied
                to the data after loading.

        :class:`spectral.image.ImageArray` is derived from both
        :class:`spectral.image.Image` and :class:`numpy.ndarray` so it supports the
        full :class:`numpy.ndarray` interface.  The returns object will have
        shape `(M,N,B)`, where `M`, `N`, and `B` are the numbers of rows,
        columns, and bands in the image.
        '''
        for k in list(kwargs.keys()):
            if k not in ('dtype', 'scale'):
                raise ValueError('Invalid keyword %s.' % str(k))
        dtype = kwargs.get('dtype', ImageArray.format)
        data = array.array(typecode('b'))
        self.fid.seek(self.offset)
        data.fromfile(self.fid, self.nrows * self.ncols *
                      self.nbands * self.sample_size)
        npArray = np.frombuffer(tobytes(data), dtype=self.dtype)
        if self.interleave == spy.BIL:
            npArray.shape = (self.nrows, self.nbands, self.ncols)
            npArray = npArray.transpose([0, 2, 1])
        elif self.interleave == spy.BSQ:
            npArray.shape = (self.nbands, self.nrows, self.ncols)
            npArray = npArray.transpose([1, 2, 0])
        else:
            npArray.shape = (self.nrows, self.ncols, self.nbands)
        npArray = npArray.astype(dtype)
        if self.scale_factor != 1 and kwargs.get('scale', True):
            npArray = npArray / float(self.scale_factor)
        imarray = ImageArray(npArray, self)
        if has_nan(imarray):
            warnings.warn('Image data contains NaN values.', NaNValueWarning)
        return imarray        

    def __getitem__(self, args):
        '''Subscripting operator that provides a numpy-like interface.
        Usage::

            x = img[i, j]
            x = img[i, j, k]

        Arguments:

            `i`, `j`, `k` (int or :class:`slice` object)

                Integer subscript indices or slice objects.

        The subscript operator emulates the :class:`numpy.ndarray` subscript
        operator, except data are read from the corresponding image file
        instead of an array object in memory.  For frequent access or when
        accessing a large fraction of the image data, consider calling
        :meth:`spectral.SpyFile.load` to load the data into an
        :meth:`spectral.image.ImageArray` object and using its subscript operator
        instead.

        Examples:

            Read the pixel at the 30th row and 51st column of the image::

                pixel = img[29, 50]

            Read the 10th band::

                band = img[:, :, 9]

            Read the first 30 bands for a square sub-region of the image::

                region = img[50:100, 50:100, :30]
        '''

        atypes = [type(a) for a in args]

        if len(args) < 2:
            raise IndexError('Too few subscript indices.')

        fix_negative_indices = self._fix_negative_indices

        if atypes[0] == atypes[1] == int and len(args) == 2:
            row = fix_negative_indices(args[0], 0)
            col = fix_negative_indices(args[1], 1)
            return self.read_pixel(row, col)
        elif len(args) == 3 and atypes[0] == atypes[1] == atypes[2] == int:
            row = fix_negative_indices(args[0], 0)
            col = fix_negative_indices(args[1], 1)
            band = fix_negative_indices(args[2], 2)
            return self.read_datum(row, col, band)
        else:
            #  At least one arg should be a slice
            if atypes[0] == slice:
                (xstart, xstop, xstep) = (args[0].start, args[0].stop,
                                          args[0].step)
                if xstart is None:
                    xstart = 0
                if xstop is None:
                    xstop = self.nrows
                if xstep is None:
                    xstep = 1
                rows = list(range(xstart, xstop, xstep))
            else:
                rows = [args[0]]
            if atypes[1] == slice:
                (ystart, ystop, ystep) = (args[1].start, args[1].stop,
                                          args[1].step)
                if ystart is None:
                    ystart = 0
                if ystop is None:
                    ystop = self.ncols
                if ystep is None:
                    ystep = 1
                cols = list(range(ystart, ystop, ystep))
            else:
                cols = [args[1]]

        if len(args) == 2 or args[2] is None:
            bands = None
        elif atypes[2] == slice:
            (zstart, zstop, zstep) = (args[2].start, args[2].stop,
                                      args[2].step)
            if zstart == zstop == zstep == None:
                bands = None
            else:
                if zstart is None:
                    zstart = 0
                if zstop is None:
                    zstop = self.nbands
                if zstep is None:
                    zstep = 1
                bands = list(range(zstart, zstop, zstep))
        elif atypes[2] == int:
            bands = [args[2]]
        else:
            # Band indices should be in a list
            bands = args[2]

        if atypes[0] == slice and xstep == 1 \
          and atypes[1] == slice and ystep == 1 \
          and (bands is None or type(bands) == list):
            xstart = fix_negative_indices(xstart, 0)
            xstop = fix_negative_indices(xstop, 0)
            ystart = fix_negative_indices(ystart, 0)
            ystop = fix_negative_indices(ystop, 0)
            bands = fix_negative_indices(bands, 2)
            return self.read_subregion((xstart, xstop), (ystart, ystop), bands)

        rows = fix_negative_indices(rows, 0)
        cols = fix_negative_indices(cols, 1)
        bands = fix_negative_indices(bands, 2)
        return self.read_subimage(rows, cols, bands)

    def _fix_negative_indices(self, indices, dim):
        if not indices:
            return indices

        dim_len = self.shape[dim]
        try:
            return [i if i >= 0 else dim_len + i
                    for i in indices]
        except:
            return indices if indices >= 0 else dim_len + indices

    def params(self):
        '''Return an object containing the SpyFile parameters.'''
        p = Image.params(self)

        p.filename = self.filename
        p.offset = self.offset
        p.byte_order = self.byte_order
        p.sample_size = self.sample_size

        return p

    def __del__(self):
        self.fid.close()


class SubImage(SpyFile):
    '''
    Represents a rectangular sub-region of a larger SpyFile object.
    '''
    def __init__(self, image, row_range, col_range):
        '''Creates a :class:`Spectral.SubImage` for a rectangular sub-region.

        Arguments:

            `image` (SpyFile):

                The image for which to define the sub-image.

            `row_range` (2-tuple):

                Integers [i, j) defining the row limits of the sub-region.

            `col_range` (2-tuple):

                Integers [i, j) defining the col limits of the sub-region.

        Returns:

            A :class:`spectral.SubImage` object providing a
            :class:`spectral.SpyFile` interface to a sub-region of the image.

        Raises:

            :class:`IndexError`

        Row and column ranges must be 2-tuples (i,j) where i >= 0 and i < j.

        '''
        if row_range[0] < 0 or \
            row_range[1] > image.nrows or \
            col_range[0] < 0 or \
                col_range[1] > image.ncols:
            raise IndexError('SubImage index out of range.')

        p = image.params()

        SpyFile.__init__(self, p, image.metadata)
        self.parent = image
        self.row_offset = row_range[0]
        self.col_offset = col_range[0]
        self.nrows = row_range[1] - row_range[0]
        self.ncols = col_range[1] - col_range[0]
        self.shape = (self.nrows, self.ncols, self.nbands)

    def read_band(self, band):
        '''Reads a single band from the image.

        Arguments:

            `band` (int):

                Index of band to read.

        Returns:

           :class:`numpy.ndarray`

                An `MxN` array of values for the specified band.
        '''
        return self.parent.read_subregion([self.row_offset,
                                           self.row_offset + self.nrows - 1],
                                          [self.col_offset,
                                           self.col_offset + self.ncols - 1],
                                          [band])

    def read_bands(self, bands):
        '''Reads multiple bands from the image.

        Arguments:

            `bands` (list of ints):

                Indices of bands to read.

        Returns:

           :class:`numpy.ndarray`

                An `MxNxL` array of values for the specified bands. `M` and `N`
                are the number of rows & columns in the image and `L` equals
                len(`bands`).
        '''
        return self.parent.read_subregion([self.row_offset,
                                           self.row_offset + self.nrows - 1],
                                          [self.col_offset,
                                           self.col_offset + self.ncols - 1],
                                          bands)

    def read_pixel(self, row, col):
        '''Reads the pixel at position (row,col) from the file.

        Arguments:

            `row`, `col` (int):

                Indices of the row & column for the pixel

        Returns:

           :class:`numpy.ndarray`

                A length-`B` array, where `B` is the number of image bands.
        '''
        return self.parent.read_pixel(row + self.row_offset,
                                      col + self.col_offset)

    def read_subimage(self, rows, cols, bands=[]):
        '''
        Reads arbitrary rows, columns, and bands from the image.

        Arguments:

            `rows` (list of ints):

                Indices of rows to read.

            `cols` (list of ints):

                Indices of columns to read.

            `bands` (list of ints):

                Optional list of bands to read.  If not specified, all bands
                are read.

        Returns:

           :class:`numpy.ndarray`

                An `MxNxL` array, where `M` = len(`rows`), `N` = len(`cols`),
                and `L` = len(bands) (or # of image bands if `bands` == None).
        '''
        return self.parent.read_subimage(list(array.array(rows) \
                                              + self.row_offset),
                                         list(array.array(cols) \
                                              + self.col_offset),
                                         bands)

    def read_subregion(self, row_bounds, col_bounds, bands=None):
        '''
        Reads a contiguous rectangular sub-region from the image.

        Arguments:

            `row_bounds` (2-tuple of ints):

                (a, b) -> Rows a through b-1 will be read.

            `col_bounds` (2-tuple of ints):

                (a, b) -> Columnss a through b-1 will be read.

            `bands` (list of ints):

                Optional list of bands to read.  If not specified, all bands
                are read.

        Returns:

           :class:`numpy.ndarray`

                An `MxNxL` array.
        '''
        return self.parent.read_subimage(list(np.array(row_bounds) \
                                              + self.row_offset),
                                         list(np.array(col_bounds) \
                                              + self.col_offset),
                                         bands)


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
    x = (np.array(list(range(nrows + 1))) * float(im.nrows) / nrows).astype(int)
    y = (np.array(list(range(ncols + 1))) * float(im.ncols) / ncols).astype(int)
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

def transform_image(transform, img):
    '''Applies a linear transform to an image.

    Arguments:

        `transform` (ndarray or LinearTransform):

            The `CxB` linear transform to apply.

        `img` (ndarray or :class:`spectral.SpyFile`):

            The `MxNxB` image to be transformed.

    Returns (ndarray or :class:spectral.spyfile.TransformedImage`):

        The transformed image.

    If `img` is an ndarray, then a `MxNxC` ndarray is returned.  If `img` is
    a :class:`spectral.SpyFile`, then a
    :class:`spectral.spyfile.TransformedImage` is returned.
    '''
    from ..algorithms.transforms import LinearTransform
    if isinstance(img, np.ndarray):
        if isinstance(transform, LinearTransform):
            return transform(img)
        ret = np.empty(img.shape[:2] + (transform.shape[0],), img.dtype)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                ret[i, j] = np.dot(transform, img[i, j])
        return ret
    else:
        return TransformedImage(transform, img)


class TransformedImage(Image):
    '''
    An image with a linear transformation applied to each pixel spectrum.
    The transformation is not applied until data is read from the image file.
    '''
    dtype = np.dtype('f4').char

    def __init__(self, transform, img):
        from ..algorithms.transforms import LinearTransform
        if not isinstance(img, Image):
            raise Exception(
                'Invalid image argument to to TransformedImage constructor.')

        if isinstance(transform, np.ndarray):
            transform = LinearTransform(transform)
        self.transform = transform

        if self.transform.dim_in not in (None, img.shape[-1]):
            raise Exception('Number of bands in image (%d) do not match the '
                            ' input dimension of the transform (%d).'
                            % (img.shape[-1], transform.dim_in))

        params = img.params()
        self.set_params(params, params.metadata)

        # If img is also a TransformedImage, then just modify the transform
        if isinstance(img, TransformedImage):
            self.transform = self.transform.chain(img.transform)
            self.image = img.image
        else:
            self.image = img
        if self.transform.dim_out is not None:
            self.shape = self.image.shape[:2] + (self.transform.dim_out,)
            self.nbands = self.transform.dim_out
        else:
            self.shape = self.image.shape
            self.nbands = self.image.nbands

    @property
    def bands(self):
        return self.image.bands

    def __getitem__(self, args):
        '''
        Get data from the image and apply the transform.
        '''
        if len(args) < 2:
            raise Exception('Must pass at least two subscript arguments')

        # Note that band indices are wrt transformed features
        if len(args) == 2 or args[2] is None:
            bands = list(range(self.nbands))
        elif type(args[2]) == slice:
            (zstart, zstop, zstep) = (args[2].start, args[2].stop,
                                      args[2].step)
            if zstart is None:
                zstart = 0
            if zstop is None:
                zstop = self.nbands
            if zstep is None:
                zstep = 1
            bands = list(range(zstart, zstop, zstep))
        elif isinstance(args[2], int):
            bands = [args[2]]
        else:
            # Band indices should be in a list
            bands = args[2]

        orig = self.image.__getitem__(args[:2])
        if len(orig.shape) == 1:
            orig = orig[np.newaxis, np.newaxis, :]
        elif len(orig.shape) == 2:
            orig = orig[np.newaxis, :]
        transformed_xy = np.zeros(orig.shape[:2] + (self.shape[2],),
                                  self.transform.dtype)
        for i in range(transformed_xy.shape[0]):
            for j in range(transformed_xy.shape[1]):
                transformed_xy[i, j] = self.transform(orig[i, j])
        # Remove unnecessary dimensions

        transformed = np.take(transformed_xy, bands, 2)

        return transformed.squeeze()

    def __str__(self):
        s = '\tTransformedImage object with output dimensions:\n'
        s += '\t# Rows:         %6d\n' % (self.nrows)
        s += '\t# Samples:      %6d\n' % (self.ncols)
        s += '\t# Bands:        %6d\n\n' % (self.shape[2])
        s += '\tThe linear transform is applied to the following image:\n\n'
        s += str(self.image)
        return s

    def read_pixel(self, row, col):
        return self.transform(self.image.read_pixel(row, col))

    def load(self):
        '''Loads all image data, transforms it, and returns an ndarray).'''
        data = self.image.load()
        return self.transform(data)

    def read_subregion(self, row_bounds, col_bounds, bands=None):
        '''
        Reads a contiguous rectangular sub-region from the image. First
        arg is a 2-tuple specifying min and max row indices.  Second arg
        specifies column min and max.  If third argument containing list
        of band indices is not given, all bands are read.
        '''
        data = self.image.read_subregion(row_bounds, col_bounds)
        xdata = self.transform(data)
        if bands:
            return np.take(xdata, bands, 2)
        else:
            return xdata

    def read_subimage(self, rows, cols, bands=None):
        '''
        Reads a sub-image from a rectangular region within the image.
        First arg is a 2-tuple specifying min and max row indices.
        Second arg specifies column min and max. If third argument
        containing list of band indices is not given, all bands are read.
        '''
        data = self.image.read_subimage(rows, cols)
        xdata = self.transform(data)
        if bands:
            return np.take(xdata, bands, 2)
        else:
            return xdata

    def read_datum(self, i, j, k):
        return self.read_pixel(i, j)[k]

    def read_bands(self, bands):
        shape = (self.image.nrows, self.image.ncols, len(bands))
        data = np.zeros(shape, float)
        for i in range(shape[0]):
            for j in range(shape[1]):
                data[i, j] = self.read_pixel(i, j)[bands]
        return data

class MemmapFile(object):
    '''Interface class for SpyFile subclasses using `numpy.memmap` objects.'''

    def _disable_memmap(self):
        '''Disables memmap and reverts to direct file reads (slower).'''
        self._memmap = None

    @property
    def using_memmap(self):
        '''Returns True if object is using a `numpy.memmap` to read data.'''
        return self._memmap is not None

    def open_memmap(self, **kwargs):
        '''Returns a new `numpy.memmap` object for image file data access.

        Keyword Arguments:

            `interleave` (str, default 'bip'):

                Specifies the shape/interleave of the returned object. Must be
                one of ['bip', 'bil', 'bsq', 'source']. If not specified, the
                memmap will be returned as 'bip'. If the interleave is
                'source', the interleave of the memmap will be the same as the
                source data file. If the number of rows, columns, and bands in
                the file are R, C, and B, the shape of the returned memmap
                array will be as follows:

                .. table::

                    ========== ===========
                    interleave array shape
                    ========== ===========
                    'bip'      (R, C, B)
                    'bil'      (R, B, C)
                    'bsq'      (B, R, C)
                    ========== ===========

            `writable` (bool, default False):

                If `writable` is True, modifying values in the returned memmap
                will result in corresponding modification to the image data
                file.
        '''        
        src_inter = {spy.BIL: 'bil',
                     spy.BIP: 'bip',
                     spy.BSQ: 'bsq'}[self.interleave]
        dst_inter = kwargs.get('interleave', 'bip').lower()
        if dst_inter not in ['bip', 'bil', 'bsq', 'source']:
            raise ValueError('Invalid interleave specified.')
        if kwargs.get('writable', False) is True:
            mode = 'r+'
        else:
            mode = 'r'
        memmap = self._open_memmap(mode)
        if dst_inter == 'source':
            dst_inter = src_inter
        if src_inter == dst_inter:
            return memmap
        else:
            return np.transpose(memmap, interleave_transpose(src_inter,
                                                             dst_inter))

    def asarray(self, writable=False):
        '''Returns an object with a standard numpy array interface.

        The function returns a numpy memmap created with the
        `open_memmap` method.

        This function is for compatibility with ImageArray objects.

        Keyword Arguments:

            `writable` (bool, default False):

                If `writable` is True, modifying values in the returned
                memmap will result in corresponding modification to the
                image data file.
        '''
        return self.open_memmap(writable=writable)

def interleave_transpose(int1, int2):
    '''Returns the 3-tuple of indices to transpose between interleaves.

    Arguments:

        `int1`, `int2` (string):

            The input and output interleaves.  Each should be one of "bil",
            "bip", or "bsq".

    Returns:

        A 3-tuple of integers that can be passed to `numpy.transpose` to
        convert and RxCxB image between the two interleaves.
    '''
    if int1.lower() not in ('bil', 'bip', 'bsq'):
        raise ValueError('Invalid interleave: %s' % str(int1))
    if int2.lower() not in ('bil', 'bip', 'bsq'):
        raise ValueError('Invalid interleave: %s' % str(int2))
    int1 = int1.lower()
    int2 = int2.lower()
    if int1 == 'bil':
        if int2 == 'bil':
            return (1, 1, 1)
        elif int2 == 'bip':
            return (0, 2, 1)
        else:
            return (1, 0, 2)
    elif int1 == 'bip':
        if int2 == 'bil':
            return (0, 2, 1)
        elif int2 == 'bip':
            return (1, 1, 1)
        else:
            return (2, 0, 1)
    else:  # bsq
        if int2 == 'bil':
            return (1, 0, 2)
        elif int2 == 'bip':
            return (1, 2, 0)
        else:
            return (1, 1, 1)
