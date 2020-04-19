'''
Generic functions for handling spectral images.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import numbers
import numpy as np

from .spectral import BandInfo

class Image(object):
    '''spectral.Image is the common base class for spectral image objects.'''

    def __init__(self, params, metadata=None):
        self.bands = BandInfo()
        self.set_params(params, metadata)

    def set_params(self, params, metadata):
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


class ImageArray(np.ndarray, Image):
    '''ImageArray is an interface to an image loaded entirely into memory.
    ImageArray objects are returned by :meth:`spectral.SpyFile.load`.
    This class inherits from both numpy.ndarray and Image, providing the
    interfaces of both classes.
    '''

    format = 'f'        # Use 4-byte floats for data arrays

    def __new__(subclass, data, spyfile):
        obj = np.asarray(data).view(subclass)
        ImageArray.__init__(obj, data, spyfile)
        return obj

    def __init__(self, data, spyfile):
        # Add param data to Image initializer
        params = spyfile.params()
        params.dtype = data.dtype
        params.swap = 0

        Image.__init__(self, params, spyfile.metadata)
        self.bands = spyfile.bands
        self.filename = spyfile.filename
        self.interleave = 2 # bip

    def __repr__(self):
        lst = np.array2string(np.asarray(self), prefix="ImageArray(")
        return "{}({}, dtype={})".format('ImageArray', lst, self.dtype.name)

    def __getitem__(self, args):
        # Duplicate the indexing behavior of SpyFile.  If args is iterable
        # with length greater than one, and if not all of the args are
        # scalars, then the scalars need to be replaced with slices.
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
            elif isinstance(arg, np.bool_):
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
        return np.ndarray.__getitem__(self, args)

    def read_band(self, i):
        '''
        For compatibility with SpyFile objects. Returns arr[:,:,i].squeeze()
        '''
        return np.asarray(self[:, :, i].squeeze())

    def read_bands(self, bands):
        '''For SpyFile compatibility. Equivlalent to arr.take(bands, 2)'''
        return np.asarray(self.take(bands, 2))

    def read_pixel(self, row, col):
        '''For SpyFile compatibility. Equivlalent to arr[row, col]'''
        return np.asarray(self[row, col])

    def read_subregion(self, row_bounds, col_bounds, bands=None):
        '''
        For SpyFile compatibility.

        Equivalent to arr[slice(*row_bounds), slice(*col_bounds), bands],
        selecting all bands if none are specified.
        '''
        if bands:
            return np.asarray(self[slice(*row_bounds),
                                   slice(*col_bounds),
                                   bands])
        else:
            return np.asarray(self[slice(*row_bounds),
                                   slice(*col_bounds)])

    def read_subimage(self, rows, cols, bands=None):
        '''
        For SpyFile compatibility.

        Equivalent to arr[rows][:, cols][:, :, bands], selecting all bands if
        none are specified.
        '''
        if bands:
            return np.asarray(self[rows][:, cols][:, :, bands])
        else:
            return np.asarray(self[rows][:, cols])

    def read_datum(self, i, j, k):
        '''For SpyFile compatibility. Equivlalent to arr[i, j, k]'''
        return np.asscalar(self[i, j, k])

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
        arr = np.asarray(self)
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
        if ((name in np.ndarray.__dict__) and
            (name not in ImageArray.__dict__)):
            return getattr(np.asarray(self), name)

        return super(ImageArray, self).__getattribute__(name)

