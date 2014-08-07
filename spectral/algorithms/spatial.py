#########################################################################
#
#   spatial.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2012-2014 Thomas Boggs
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
Functions over spatial regions of images.
'''

__all__ = ['apply_windowed_function']

import numpy as np

def get_window_bounds(nrows, ncols, height, width, i, j):
    '''Returns boundaries of an image window centered on a specified pixel.

    Arguments:

        `nrows` (int):

            Total number of rows in the image

        `ncols` (int):

            Total number of columns in the image

        `height` (int):

            Height of the desired window (in pixels)

        `width` (int):

            Width of the desired window (in pixels)

        `i` (int):

            Row index of the pixel

        `j` (int):

            Column index of the pixel

    Return value:

        A 4-tuple of ints of the form

            (row_start, row_stop, col_start, col_stop).
            
    The dimensions of the window will always be (`height`, `width`). For
    pixels near the border of the image where there are insufficient pixels
    between the specified pixel and image border, the window will be flush
    against the border of the image and the pixel position will be offset
    from the center of the widow.

    For an alternate function that clips window pixels near the border of the
    image, see `get_clipped_window_bounds`.
    '''
    if height > nrows or width > ncols:
        raise ValueError('Window size is too large for image dimensions.')

    rmin = i - height / 2
    rmax = rmin + height
    if rmin < 0:
        rmax = height
        rmin = 0
    elif rmax > nrows:
        rmax = nrows
        rmin = nrows - height

    cmin = j - width / 2
    cmax = cmin + width
    if cmin < 0:
        cmax = width
        cmin = 0
    elif cmax > ncols:
        cmax = ncols
        cmin = ncols - width

    return (rmin, rmax, cmin, cmax)

def get_window_bounds_clipped(nrows, ncols, height, width, i, j):
    '''Returns boundaries of an image window centered on a specified pixel.

    Arguments:

        `nrows` (int):

            Total number of rows in the image

        `ncols` (int):

            Total number of columns in the image

        `height` (int):

            Height of the desired window (in pixels)

        `width` (int):

            Width of the desired window (in pixels)

        `i` (int):

            Row index of the pixel

        `j` (int):

            Column index of the pixel

    Return value:

        A 4-tuple of ints of the form

            (row_start, row_stop, col_start, col_stop).
            
    Near the boder of the image where there are insufficient pixels between
    the specified pixel and the image border, the window will be clipped.

    For an alternate function that always returns a window with dimensions
    (`width`, `height`), see `get_window_bounds`.
    '''
    if height > nrows or width > ncols:
        raise ValueError('Window size is too large for image dimensions.')

    rmin = i - height / 2
    rmax = rmin + height
    if rmin < 0:
        rmin = 0
    elif rmax > nrows:
        rmax = nrows

    cmin = j - width / 2
    cmax = cmin + width
    if cmin < 0:
        cmin = 0
    elif cmax > ncols:
        cmax = ncols

    return (rmin, rmax, cmin, cmax)

def apply_windowed_function(func, image, height, width,
                            rstart=0, rstop=-1, cstart=0, cstop=-1,
                            border='shift', dtype=None ):
    '''Applies a function over a rolling spatial window.
    
    Arguments:

        `func` (callable):

            The function to apply. For an `N`-band image, this function must
            accept as input an ndarray with shape `(height, width, N)`. For
            some values of the `border` argument, the first two dimensions of
            the functions input may be smaller.

        `image` (`SpyFile` or np.ndarray):

            The image on which the apply `func` with the specified window.

        `height` (int):

            The height of the rolling window in pixels
            
        `width` (int):

            The width of the rolling window in pixels

        `rstart` (int, default 0):

            Index of the first row for which the function is applied. If not
            specified, start on the first row.

        `rstop` (int, default -1):

            Index of the row at which iteration stops. If not specified, stop
            after the last row.
 
        `cstart` (int, default 0):

            Index of the first column for which the function is applied. If not
            specified, start on the first columnn.

        `cstop` (int, default -1):

            Index of the column at which iteration stops. If not specified,
            stop after the last row.

        `border` (string, default "shift"):

            Indicates how to handles windows near the edge of the window. If
            the value is "shift", the window dimensions will alway be
            `(width, height)` but near the image border the pixel being
            iterated will be offset from the center of the window. If set to
            "clip", window regions falling outside the image border will be
            clipped and the window dimension will be reduced.

        `dtype` (np.dtype):

            Optional dtype for the output.

    Return value:

        Returns an np.ndarray with shape corresponding to the row and column
        start/stop indices and shape of `func` output.

    Examples:
    ---------

    To produce a new image that is a 3x3 pixel average of the input image:

    >>> f = lambda X: np.mean(X.reshape((-1, X.shape[-1])), axis=0)
    >>> image_3x3 = apply_windowed_function(f, image, 3, 3)
   '''
    if border == 'shift':
        get_window = get_window_bounds
    elif border == 'clip':
        get_window = get_window_bounds_clipped
    else:
        raise ValueError('Unrecognized border option.')

    if rstop < 0:
        rstop = image.shape[0] + 1 + rstop
    if cstop < 0:
        cstop = image.shape[1] + 1 + cstop

    (nrows, ncols) = image.shape[:2]

    # Call the function once to get output shape and dtype
    (r0, r1, c0, c1) = get_window(nrows, ncols, height, width, rstart, cstart)
    y = func(image[r0:r1, c0:c1])
    if dtype is None:
        dtype = np.array(y).dtype
    out = np.empty((rstop - rstart, cstop - cstart) + np.shape(y), dtype=dtype)

    for i in range(rstart, rstop):
        for j in range(cstart, cstop):
            (r0, r1, c0, c1) = get_window(nrows, ncols, height, width, i, j)
            out[i - rstart, j - cstart] = func(image[r0:r1, c0:c1])
    return out
            

def inner_outer_window_mask_creator(image_shape, window):
    '''Returns a function to give  inner/outer windows.

    Arguments:

        `image_shape` (tuple of integers):

            Specifies the dimensions of the image for which windows are to be
            produced. Only the first two dimensions (rows, columns) is used.

        `window` (2-tuple of integers):

            Specifies the sizes of the inner & outer windows. Both values
            must be odd integers.

    Return value:

        A function that accepts the following arguments:

            `i` (int):

                Row index of pixel for which to generate the mask
                
            `j` (int):

                Row index of pixel for which to generate the mask

            `gen_mask` (bool, default False):

                A boolean flag indicating whether to return a boolean mask of
                shape (window[1], window[1]), indicating which pixels in the
                window should be used for background statistics calculations.

        If `gen_mask` is False, the return value is a 2-tuple of 4-tuples,
        where the 2-tuples specify the start/stop row/col indices for the
        inner and outer windows, respectively. Each of the 4-tuples is of the
        form (row_start, row_stop, col_start, col_stop).

        If `gen_mask` is True, a third element is added the tuple, which is
        the boolean mask for the inner/outer window.
    '''
    (R, C) = image_shape[:2]
    (R_in, R_out) = window
    assert(R_in % 2 + R_out % 2 == 2)
    (a, b) = [(x - 1) / 2 for x in window]
    def create_mask(i, j, gen_mask=False):
        inner_imin = max(i - a, 0)
        inner_imax = min(i + a + 1, R)
        inner_jmin = max(j - a, 0)
        inner_jmax = min(j + a + 1, C)
        outer_imin = max(i - b, 0)
        outer_imax = min(outer_imin + R_out, R)
        if outer_imax == R:
            outer_imin = R - R_out
        outer_jmin = max(j - b, 0)
        outer_jmax = min(outer_jmin + R_out, C)
        if outer_jmax == C:
            outer_jmin = C - R_out
        inner = (inner_imin, inner_imax, inner_jmin, inner_jmax)
        outer = (outer_imin, outer_imax, outer_jmin, outer_jmax)
        if not gen_mask:
            return (inner, outer)
        mask = np.zeros((R_out, R_out), dtype=np.bool)
        mask[inner_imin - outer_imin : inner_imax - outer_imin,
             inner_jmin - outer_jmin : inner_jmax - outer_jmin] = True
        return (inner, outer, mask)
    return create_mask
