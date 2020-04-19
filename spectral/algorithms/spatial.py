'''
Functions over spatial regions of images.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = ['map_window', 'map_outer_window_stats', 'map_class_ids',
           'map_classes']

import itertools
import numpy as np

import spectral as spy
from .algorithms import GaussianStats, iterator_ij

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
    image, see `get_window_bounds_clipped`.
    '''
    if height > nrows or width > ncols:
        raise ValueError('Window size is too large for image dimensions.')

    rmin = i - height // 2
    rmax = rmin + height
    if rmin < 0:
        rmax = height
        rmin = 0
    elif rmax > nrows:
        rmax = nrows
        rmin = nrows - height

    cmin = j - width // 2
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

    rmin = i - height // 2
    rmax = rmin + height
    if rmin < 0:
        rmin = 0
    elif rmax > nrows:
        rmax = nrows

    cmin = j - width // 2
    cmax = cmin + width
    if cmin < 0:
        cmin = 0
    elif cmax > ncols:
        cmax = ncols

    return (rmin, rmax, cmin, cmax)

def map_window(func, image, window, rslice=(None,), cslice=(None,),
               border='shift', dtype=None):
    '''Applies a function over a rolling spatial window.
    
    Arguments:

        `func` (callable):

            The function to apply. This function must accept two inputs:

            `X` (ndarray):

                The image data corresponding to the spatial window for the
                current pixel being evaluated. `X` will have shape
                `window + (N,)`, where `N` is the number of bands in the image.
                For pixels near the border of the image, the first two
                dimensions of `X` may be smaller if `border` is set to "clip".

            `ij` (2-tuple of integers):

                Indicates the row/column of the current pixel within the
                window. For `window` with even dimensions or for pixels near
                the image border, this may not correspond to the center pixel
                in the window.
    
        `image` (`SpyFile` or np.ndarray):

            The image on which the apply `func` with the specified window.

        `window` (int or 2-tuple of ints):

            The size of the window, in pixels. If this value is an integer,
            the height and width of the window will both be set to the value.
            Otherwise, `window` should be a tuple of the form (height, width).
            
        `rslice` (tuple):

            Tuple of `slice` parameters specifying at which rows the function
            should be applied. If not provided, `func` is applied to all rows.

        `cslice` (tuple):

            Tuple of `slice` parameters specifying at which columns the
            function should be applied. If not provided, `func` is applied to
            all columns.

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

    >>> f = lambda X, ij: np.mean(X.reshape((-1, X.shape[-1])), axis=0)
    >>> image_3x3 = map_window(f, image, 3)

    Perform a 5x5 pixel average but only retain values at every fifth row and
    column (i.e., simulate an image at one fifth resolution):

    >>> image.shape
    (145, 145, 220)
    >>> image_5x5 = map_window(f, image, 5, (2, -2, 5), (2, -2, 5))
    >>> image_5x5.shape
    (29, 29, 220)
    '''
    if isinstance(window, (list, tuple)):
        (height, width) = window[:]
    else:
        (height, width) = (window, window)

    if border == 'shift':
        get_window = get_window_bounds
    elif border == 'clip':
        get_window = get_window_bounds_clipped
    else:
        raise ValueError('Unrecognized border option.')

    (nrows, ncols) = image.shape[:2]

    # Row/Col indices at which to apply the windowed function
    rvals = list(range(*slice(*rslice).indices(nrows)))
    cvals = list(range(*slice(*cslice).indices(ncols)))

    def get_val(i, j):
        (r0, r1, c0, c1) = get_window(nrows, ncols, height, width, i, j)
        return func(image[r0:r1, c0:c1],
                    (i - r0, j - c0)).astype(dtype)

    return np.array([[get_val(r, c) for c in cvals]
                     for r in rvals]).astype(dtype)

def map_outer_window_stats(func, image, inner, outer, dim_out=1, cov=None,
                           dtype=None, rslice=(None,), cslice=(None,)):
    '''Maps a function accepting `GaussianStats` over a rolling spatial window.
    
    Arguments:

        `func` (callable):

            A callable object that will be applied to each pixel when the
            __call__ method is called for this object. The __call__ method
            of `func` must accept two arguments:

                - `X` (`GaussianStats`):

                    The Gaussian statistics computed from pixels in the outer
                    window (excluding the inner window).

                - `v` (ndarray):

                    An ndarray representing the pixel for which the window
                    was produced.

        `image` (`SpyFile` or np.ndarray):

            The image on which the apply `func` with the specified window.

        `inner` (int or 2-tuple of ints):

            The size of the inner window, in pixels. If this value is an integer,
            the height and width of the window will both be set to the given value.
            Otherwise, `inner` should be a tuple of the form (height, width).
            All pixels within the inner window are excluded from statistics
            computed for the associated pixel.
            
        `outer` (int or 2-tuple of ints):

            The size of the outer window, in pixels. If this value is an integer,
            the height and width of the window will both be set to the given value.
            Otherwise, `outer` should be a tuple of the form (height, width).
            All pixels in the outer window (but not in the inner window) are
            used to compute statistics for the associated pixel.
            
        `rslice` (tuple):

            Tuple of `slice` parameters specifying at which rows the function
            should be applied. If not provided, `func` is applied to all rows.

        `cslice` (tuple):

            Tuple of `slice` parameters specifying at which columns the
            function should be applied. If not provided, `func` is applied to
            all columns.

        `dtype` (np.dtype):

            Optional dtype for the output.

    Return value:

        Returns an np.ndarray whose elements are the result of mapping `func`
        to the pixels and associated window stats.

    Examples:
    ---------

    To create an RX anomaly detector with a 3x3 pixel inner window and 17x17
    outer window (note that `spectral.rx` already does this):

    >>> def mahalanobis(bg, x):
    ...     return (x - bg.mean).dot(bg.inv_cov).dot(x - bg.mean)
    ...
    >>> rx_scores = map_outer_window_stats(mahalanobis, image, 3, 17)

    '''
    mapper = WindowedGaussianBackgroundMapper(inner, outer, func, cov, dim_out,
                                              dtype)
    return mapper(image, rslice, cslice)

class WindowedGaussianBackgroundMapper(object):
    '''A class for procucing window statistics with an inner exclusion window.
    '''
    def __init__(self, inner, outer, function=None, cov=None, dim_out=None,
                 dtype=None):
        '''Creates a detector with the given inner/outer window.

        Arguments:

            `inner` (integer or 2-tuple of integers):

                Width and heigth of inner window, in pixels.

            `outer` (integer or 2-tuple of integers):

                Width and heigth of outer window, in pixels. Dimensions must
                be greater than inner window

            `function` (callable object):

                A callable object that will be applied to each pixel when the
                __call__ method is called for this object. The __call__ method
                of `function` must accept two arguments:

                    - A `GaussianStats` object.

                    - An ndarray representing the pixel for which the
                      were computed.

            `cov` (ndarray):

                An optional covariance to use. If this parameter is given,
                `cov` will be used for all RX calculations (background
                covariance will not be recomputed in each window). Only the
                background mean will be recomputed in each window).

            `dim_out` (int):

                The dimensionality of the output of `function` when called on
                a pixel spectrum. If this value is not specified, `function`
                will be checked to see if it has a `dim_out` member. If it
                does not, `dim_out` will be assumed to be 1.

            `dtype`:

                Optional dtype for the output array. If not specified,
                np.float32 is used.
        '''
        if isinstance(inner, (list, tuple)):
            self.inner = inner[:]
        else:
            self.inner = (inner, inner)
        if isinstance(outer, (list, tuple)):
            self.outer = outer[:]
        else:
            self.outer = (outer, outer)
        self.callable = function
        self.cov = cov
        self.dim_out = dim_out
        self.create_mask = None
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = np.float32

    def __call__(self, image, rslice=(None,), cslice=(None,)):
        '''Applies the objects callable function to the image data.

        Arguments:

            `image` (numpy.ndarray):

                An image with shape (R, C, B).

            `rslice` (tuple):

                Tuple of `slice` parameters specifying at which rows the function
                should be applied. If not provided, `func` is applied to all rows.

            `cslice` (tuple):

                Tuple of `slice` parameters specifying at which columns the
                function should be applied. If not provided, `func` is applied to
                all columns.

        Returns numpy.ndarray:

            An array whose elements correspond to the outputs from the
            object's callable function.
        '''
        (R, C, B) = image.shape
        (row_border, col_border) = [x // 2 for x in self.outer]

        if self.dim_out is not None:
            dim_out = self.dim_out
        elif hasattr(self.callable, 'dim_out') and \
          self.callable.dim_out is not None:
            dim_out = self.callable.dim_out
        else:
            dim_out = 1

        # Row/Col indices at which to apply the windowed function
        rvals = list(range(*slice(*rslice).indices(R)))
        cvals = list(range(*slice(*cslice).indices(C)))

        nrows_out = len(rvals)
        ncols_out = len(cvals)

        if dim_out > 1:
            x = np.ones((nrows_out, ncols_out, dim_out),
                        dtype=np.float32) * -1.0
        else:
            x = np.ones((nrows_out, ncols_out), dtype=self.dtype) * -1.0

        npixels = self.outer[0] * self.outer[1] - self.inner[0] * self.inner[1]
        if self.cov is None and npixels < B:
            raise ValueError('Window size provides too few samples for ' \
                             'image data dimensionality.')

        if self.create_mask is not None:
            create_mask = self.create_mask
        else:
            create_mask = inner_outer_window_mask_creator(image.shape,
                                                          self.inner,
                                                          self.outer)

        interior_mask = create_mask(R // 2, C // 2, True)[2].ravel()
        interior_indices = np.argwhere(interior_mask == 0).squeeze()

        (i_interior_start, i_interior_stop) = (row_border, R - row_border)
        (j_interior_start, j_interior_stop) = (col_border, C - col_border)

        status = spy._status
        status.display_percentage('Processing image: ')
        if self.cov is not None:
            # Since we already have the covariance, just use np.mean to get
            # means of the inner window and outer (including the inner), then
            # use those to calculate the mean of the outer window alone.
            background = GaussianStats(cov=self.cov)
            for i in range(nrows_out):
                for j in range(ncols_out):
                    (inner, outer) = create_mask(rvals[i], cvals[j], False)
                    N_in = (inner[1] - inner[0]) * (inner[3] - inner[2])
                    N_tot = (outer[1] - outer[0]) * (outer[3] - outer[2])
                    mean_out = np.mean(image[outer[0]: outer[1],
                                             outer[2]: outer[3]].reshape(-1, B),
                                             axis=0)
                    mean_in = np.mean(image[outer[0]: outer[1],
                                            outer[2]: outer[3]].reshape(-1, B),
                                            axis=0)
                    mean = mean_out * (float(N_tot) / (N_tot - N_in)) - \
                           mean_in * (float(N_in) / (N_tot - N_in))
                    background.mean = mean
                    x[i, j] = self.callable(background,
                                            image[rvals[i], cvals[j]])
                if i % (nrows_out // 10) == 0:
                    status.update_percentage(100. * i // nrows_out)
        else:
            # Need to calculate both the mean and covariance for the outer
            # window (without the inner).
            (h, w) = self.outer[:]
            for i in range(nrows_out):
                ii = rvals[i] - h // 2
                for j in range(ncols_out):
                    jj = cvals[j] - w // 2
                    if i_interior_start <= rvals[i] < i_interior_stop and \
                        j_interior_start <= cvals[j] < j_interior_stop:
                        X = image[ii : ii + h, jj : jj + w, :]
                        indices = interior_indices
                    else:
                        (inner, (i0, i1, j0, j1), mask) = \
                          create_mask(rvals[i], cvals[j], True)
                        indices = np.argwhere(mask.ravel() == 0).squeeze()
                        X = image[i0 : i1, j0 : j1, :]
                    X = np.take(X.reshape((-1, B)), indices, axis=0)
                    mean = np.mean(X, axis=0)
                    cov = np.cov(X, rowvar=False)
                    background = GaussianStats(mean, cov)
                    x[i, j] = self.callable(background,
                                            image[rvals[i], cvals[j]])
                if i % (nrows_out // 10) == 0:
                    status.update_percentage(100. * i / nrows_out)

        status.end_percentage()
        return x

def inner_outer_window_mask_creator(image_shape, inner, outer):
    '''Returns a function to give  inner/outer windows.

    Arguments:

        `image_shape` (tuple of integers):

            Specifies the dimensions of the image for which windows are to be
            produced. Only the first two dimensions (rows, columns) is used.

        `inner` (int or 2-tuple of integers):

            Height and width of the inner window, in pixels.

        `outer` (int or 2-tuple of integers):

            Height and width of the outer window, in pixels.

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
    if isinstance(inner, (list, tuple)):
        (hi, wi) = inner[:]
    else:
        (hi, wi) = (inner, inner)
    if isinstance(outer, (list, tuple)):
        (ho, wo) = outer[:]
    else:
        (ho, wo) = (outer, outer)

    if wi > wo or hi > ho:
        raise ValueError('Inner window dimensions must be smaller than outer.')
    
    (ai, bi) = (hi // 2, wi // 2)
    (ao, bo) = (ho // 2, wo // 2)

    def create_mask(i, j, gen_mask=False):

        # Inner window
        inner_imin = i - ai
        inner_imax = inner_imin + hi
        if inner_imin < 0:
            inner_imax = hi
            inner_imin = 0
        elif inner_imax > R:
            inner_imax = R
            inner_imin = R - hi
        
        inner_jmin = j - bi
        inner_jmax = inner_jmin + wi
        if inner_jmin < 0:
            inner_jmax = wi
            inner_jmin = 0
        elif inner_jmax > C:
            inner_jmax = C
            inner_jmin = C - wi
        
        # Outer window
        outer_imin = i - ao
        outer_imax = outer_imin + ho
        if outer_imin < 0:
            outer_imax = ho
            outer_imin = 0
        elif outer_imax > R:
            outer_imax = R
            outer_imin = R - ho
        
        outer_jmin = j - bo
        outer_jmax = outer_jmin + wo
        if outer_jmin < 0:
            outer_jmax = wo
            outer_jmin = 0
        elif outer_jmax > C:
            outer_jmax = C
            outer_jmin = C - wo
        
        inner = (inner_imin, inner_imax, inner_jmin, inner_jmax)
        outer = (outer_imin, outer_imax, outer_jmin, outer_jmax)
        if not gen_mask:
            return (inner, outer)
        mask = np.zeros((ho, wo), dtype=np.bool)
        mask[inner_imin - outer_imin : inner_imax - outer_imin,
             inner_jmin - outer_jmin : inner_jmax - outer_jmin] = True
        return (inner, outer, mask)
    return create_mask

def map_class_ids(src_class_image, dest_class_image, unlabeled=None):
    '''Create a mapping between class labels in two classification images.

    Running a classification algorithm (particularly an unsupervised one)
    multiple times on the same image can yield similar results but with
    different class labels (indices) for the same classes. This function
    produces a mapping of class indices from one classification image to
    another by finding class indices that share the most pixels between the
    two classification images.

    Arguments:

        `src_class_image` (ndarray):

            An MxN integer array of class indices. The indices in this array
            will be mapped to indices in `dest_class_image`.
    
        `dest_class_image` (ndarray):

            An MxN integer array of class indices.

        `unlabeled` (int or array of ints):

            If this argument is provided, all pixels (in both images) will be
            ignored when counting coincident pixels to determine the mapping.
            If mapping a classification image to a ground truth image that has
            a labeled background value, set `unlabeled` to that value.

    Return Value:

        A dictionary whose keys are class indices from `src_class_image` and
        whose values are class indices from `dest_class_image`.

    .. seealso::

       :func:`map_classes`
    '''
    src_ids = list(set(src_class_image.ravel()))
    dest_ids = list(set(dest_class_image.ravel()))
    cmap = {}
    if unlabeled is not None:
        if isinstance(unlabeled, int):
            unlabeled = [unlabeled]
        for i in unlabeled:
            if i in src_ids:
                src_ids.remove(i)
                cmap[i] = i
            if i in dest_ids:
                dest_ids.remove(i)
    else:
        unlabeled = []
    N_src = len(src_ids)
    N_dest = len(dest_ids)

    # Create matrix of coincidence counts between classes in src and dest.
    matches = np.zeros((N_src, N_dest), np.uint16)
    for i in range(N_src):
        src_is_i = (src_class_image == src_ids[i])
        for j in range(N_dest):
            matches[i, j] = np.sum(np.logical_and(src_is_i,
                                                  dest_class_image == dest_ids[j]))

    unmapped = set(src_ids)
    dest_available = set(dest_ids)
    while len(unmapped) > 0:
        (i, j) = tuple(np.argwhere(matches == np.max(matches))[0])
        mmax = matches[i, j]
        if mmax == 0:
            # Nothing left to map. Pick unused indices from dest_class_image
            for (old, new) in zip(sorted(unmapped), sorted(dest_available)):
                cmap[old] = new
                unmapped.remove(old)
                dest_available.remove(new)
            for old in unmapped:
                # The list of target classes has been exhausted. Pick the
                # smallest dest value that isn't already used.
                def next_id():
                    for ii in itertools.count():
                        if ii not in unlabeled and ii not in cmap.values():
                            return ii
                cmap[old] = next_id()
            break
        cmap[src_ids[i]] = dest_ids[j]
        unmapped.remove(src_ids[i])
        dest_available.remove(dest_ids[j])
        matches[i, :] = 0
        matches[:, j] = 0
    return cmap

def map_classes(class_image, class_id_map, allow_unmapped=False):
    '''Modifies class indices according to a class index mapping.

    Arguments:

        `class_image`: (ndarray):

            An MxN array of integer class indices.

        `class_id_map`: (dict):

            A dict whose keys are indices from `class_image` and whose values
            are new values for the corresponding indices. This value is
            usually the output of :func:`map_class_ids`.

        `allow_unmapped` (bool, default False):

            A flag indicating whether class indices can appear in `class_image`
            without a corresponding key in `class_id_map`. If this value is
            False and an index in the image is found without a mapping key,
            a :class:`ValueError` is raised. If True, the unmapped index will
            appear unmodified in the output image.

    Return Value:

        An integer-valued ndarray with same shape as `class_image`

    Example:

        >>> m = spy.map_class_ids(result, gt, unlabeled=0)
        >>> result_mapped = spy.map_classes(result, m)

    .. seealso::

       :func:`map_class_ids`
    '''
    if not allow_unmapped  \
      and not set(class_id_map.keys()).issuperset(set(class_image.ravel())):
        raise ValueError('`src` has class values with no mapping key')
    mapped = np.array(class_image)
    for (i, j) in class_id_map.items():
        mapped[class_image == i] = j
    return mapped

def expand_binary_mask_for_window(mask, height, width):
    '''Returns a new mask including window around each pixel in source mask.

    Arguments:

        `mask` (2D ndarray):

            An ndarray whose non-zero elements define a mask.

        `height` (int):

            Height of the window.

        `width` (int):

            Width of the window

    Returns a new mask of ones and zeros with same shape as `mask`. For each
    non-zero element in mask, the returned mask will contain a value of one
    for all pixels in the `height`x`width` window about the pixel and zeros
    elsewhere.
    '''
    m = np.zeros_like(mask)
    (mask_height, mask_width) = mask.shape
    for (i, j) in iterator_ij(mask):
        (r0, r1, c0, c1) = get_window_bounds_clipped(mask_height, mask_width,
                                                     height, width, i, j)
        m[r0:r1, c0:c1] = 1
    return m
