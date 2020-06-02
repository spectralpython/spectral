'''
Continuum and continuum removal.

Continuum is defined as convex hull of spectrum.
Continuum is removed from spectra by dividing spectra by its continuum.
That results in values between 0 and 1, where absorption bands are expressed as
drops below 1. It is usefull for comparing and classification based on
absorption bands and indifferent to scale.

References:
    Clark, R.N. and Roush, L. (1984) Reflectance Spectroscopy Quantitative Analysis
    Techniques for Remote Sensing Applications. Journal of Geophysical Research,
    89, 6329-6340. http://dx.doi.org/10.1029/JB089iB07p06329

    Jiwei Bai, et al., "Classification methods of the hyperspectralimage based
    on the continuum-removed," Proc. SPIE 4897, Multispectral and Hyperspectral
    Remote Sensing Instruments and Applications, (16 June 2003);
    doi: 10.1117/12.466729
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np

import spectral as spy
from ..utilities.errors import has_nan, NaNValueError

def _find_indices_in_range(spectrum, bands, indices, ind_fill, ibegin, iend):
    iendi = iend - 1

    # We search for maximum, but not from the x axis.
    # We search for maximum w.r.t to axis represented by line connecting
    # first and last point (of this iteration).

    # First find normal to new axis. Swap x and y, and negate new x.
    # If we negate x instead of y, normal will always point upward.
    naxis_y = bands[iendi] - bands[ibegin]
    naxis_x = spectrum[ibegin] - spectrum[iendi]

    # Don't literally compute distance from the axis. Rather, calculate dot products
    # of points with the normal, and find the largest. The largest dot product (it does not have to be positive)
    # is the one that goes more in the direction of normal than others. To get the distance,
    # we could divide each dot product by norm/length of the normal. But that is constant,
    # and does not effect which one is maximum.
    # Note that here we include first point of the range, but not last.
    imax = np.argmax(bands[ibegin:iendi] * naxis_x + spectrum[ibegin:iendi] * naxis_y) + ibegin

    # If first point is maximum, then all others are "below" the axis,
    # which means this is concave region.
    if imax == ibegin:
        return ind_fill

    # Repeat same procedure on the left side, if there are enough points left.
    # At least 1 is required between first and last point in range.
    if imax > ibegin + 1:
        ind_fill = _find_indices_in_range(spectrum, bands, indices, ind_fill, ibegin, imax + 1)

    # Push middle index.
    indices[ind_fill] = imax
    ind_fill += 1

    # Repeat for the right side.
    if imax < iend - 2:
        ind_fill =_find_indices_in_range(spectrum, bands, indices, ind_fill, imax, iend)

    return ind_fill

def _find_continuum_points_recursive(spectrum, bands, indices):
    n = len(spectrum)
    indices[0] = 0
    ind_fill = 1

    ind_fill = _find_indices_in_range(spectrum, bands, indices, ind_fill, 0, n)
    indices[ind_fill] = n - 1
    indices = indices[:ind_fill + 1]

    return (bands[indices], spectrum[indices])

def _process_continuum(spectra, bands, remove_continuum, out):
    if not isinstance(spectra, np.ndarray):
        raise TypeError('Expected spectra to be a numpy.ndarray.')
    if not isinstance(bands, np.ndarray):
        raise TypeError('Expected spectra to be a numpy.ndarray.')
    if out is not None and not isinstance(out, np.ndarray):
        raise TypeError('Expected out to be a numpy.ndarray or None.')
    if len(spectra.shape) not in (1, 2, 3):
        raise ValueError('Expected spectra to be 1d, 2d, or 3d array.')
    if len(bands.shape) != 1:
        raise ValueError('Expected bands to be 1d array.')
    if out is not None and not np.array_equal(out.shape, spectra.shape):
        raise ValueError('Expected out to be same shape as spectra.')

    out = np.empty_like(spectra) if out is None else out

    # In case we remove continuum, always devide out by continuum,
    # to avoid creating additional temporary array.
    if spectra is not out and remove_continuum:
        out[:] = spectra[:]

    original_shape = spectra.shape
    nbands = original_shape[-1]

    interp = np.interp
    indices = np.empty(nbands, np.int64)

    if len(spectra.shape) == 1:
        points = _find_continuum_points_recursive(spectra, bands, indices)
        continuum = interp(bands, points[0], points[1])
        if remove_continuum:
            out /= continuum
        else:
            out[:] = continuum
    elif len(spectra.shape) == 2:
        for i in range(spectra.shape[0]):
            points = _find_continuum_points_recursive(spectra[i], bands, indices)
            continuum = interp(bands, points[0], points[1])
            if remove_continuum:
                out[i, :] /= continuum
            else:
                out[i, :] = continuum
    else:
        for i in range(spectra.shape[0]):
            for j in range(spectra.shape[1]):
                points = _find_continuum_points_recursive(spectra[i, j], bands, indices)
                continuum = interp(bands, points[0], points[1])
                if remove_continuum:
                    out[i, j, :] /= continuum
                else:
                    out[i, j, :] = continuum

    return out

def continuum_points(spectrum, bands):
    '''Returns points of spectra that belong to it's continuum.

    Arguments:

        `spectrum` (:class:`numpy.ndarray`)
        
            1d :class:`numpy.ndarray` holding spectral signature.

        `bands` (:class:`numpy.ndarray`):

            1d :class:`numpy.ndarray`, holding band values of spectra.
            Length of `bands` should be the same as `spectrum`.

    Returns:
    
        2-tuple, with each element being :class:`numpy.ndarray`.
        First element contains reflectance values of points that belong to
        continuum. Second element contains corresponding bands.
        By applying linear interpolation to this data as x and y, we get
        continuum of spectrum. However this function is particularly useful to
        applying other interpolations or any other processing on these points.
    '''
    if not isinstance(spectra, np.ndarray):
        raise TypeError('Expected spectra to be a numpy.ndarray.')
    if not isinstance(bands, np.ndarray):
        raise TypeError('Expected bands to be a numpy.ndarray.')
    if len(spectra.shape) != 1:
        raise ValueError('Expected spectra to be 1d array.')
    if len(bands.shape) != 1:
        raise ValueError('Expected bands to be 1d array.')

    indices = np.empty_like(spectrum, dtype='int64')
    return _find_continuum_points_recursive(spectrum, bands, indices)

def spectral_continuum(spectra, bands, out = None):
    '''Returns continua of spectra.
    Continuum is defined as convex hull of spectra.

    Arguments:

        `spectra` (:class:`numpy.ndarray`)
        
            Can be 1d, 2d or 3d :class:`numpy.ndarray`, where last dimension
            holds individual spectra.

        `bands` (:class:`numpy.ndarray`):

            1d :class:`numpy.ndarray`, holding band values of spectra.
            Length of `bands` should be the same as last dimension of `spectra`.

        `out` (:class:`numpy.ndarray`, default None):

            If provided, it must have same type and same shape as `spectra`,
            and it will hold the result, and will be returned as result of this
            function.

    Returns:
    
        A :class:`numpy.ndarray` of continua for each spectrum in spectra.
        It same type and shape as spectra. If `out` is provided, `out` will be
        returned.
    '''
    return _process_continuum(spectra, bands, False, out)

def remove_continuum(spectra, bands, out = None):
    '''Returns spectra with continuum removed.
    Continuum is defined as convex hull of spectra. Continuum is removed from
    spectra by deviding spectra by its continuum.

    Arguments:

        `spectra` (:class:`numpy.ndarray`)
        
            Can be 1d, 2d or 3d :class:`numpy.ndarray`, where last dimension
            holds individual spectra.

        `bands` (:class:`numpy.ndarray`):

            1d :class:`numpy.ndarray`, holding band values of spectra.
            Length of `bands` should be the same as last dimension of `spectra`.

        `out` (:class:`numpy.ndarray`, default None):

            If provided, it must have type `np.float64` and same shape as
            `spectra`, and it will hold the result, and will be returned as
            result of this function.

    Returns:
    
        A :class:`numpy.ndarray` of continua for in spectrum in spectra.
        It type `np.float64` and same shape as spectra. If `out` is provided,
        `out` will be returned.
    '''
    if out is not None and out.dtype != np.float64:
        raise ValueError('Expected out to have dtype float64. '
                        'Results of continuum removal are floating point numbers.')
    return _process_continuum(spectra, bands, True, out)