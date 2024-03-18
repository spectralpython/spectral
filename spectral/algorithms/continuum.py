'''
Continuum and continuum removal.

Continuum is defined as convex hull of spectrum.
Continuum is removed from spectra by dividing spectra by its continuum.
That results in values between 0 and 1, where absorption bands are expressed as
drops below 1. It is useful for comparing and classification based on
absorption bands and indifferent to scale.

References:
    Clark, R.N. and Roush, L. (1984) Reflectance Spectroscopy Quantitative Analysis
    Techniques for Remote Sensing Applications. Journal of Geophysical Research,
    89, 6329-6340. http://dx.doi.org/10.1029/JB089iB07p06329

    Jiwei Bai, et al., "Classification methods of the hyperspectralimage based
    on the continuum-removed," Proc. SPIE 4897, Multispectral and Hyperspectral
    Remote Sensing Instruments and Applications, (16 June 2003);
    doi: 10.1117/12.466729

    Lehnert, Lukas & Meyer, Hanna & Obermeier, Wolfgang & Silva, Brenner & Regeling,
    Bianca & Thies, Boris & Bendix, Jorg. (2019). Hyperspectral Data Analysis in R:
    The hsdar Package. Journal of statistical software. 89. 1-23. 10.18637/jss.v089.i12.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


def _segment_concave_region(spectrum, bands, indices, ind_fill, ibegin, iend):
    # Here we don't search for local maxima w.r.t. line that connects ends of this region.
    # That is behavior of the hsdar. It also makes more sense in the context of
    # hyperspectral image analysis. We are already not convex, and we can't
    # include all points that make result quasi-convex, since there will be too
    # many of them, often right one after another. However, filtering local maxima,
    # below, will make result quasi-convex.
    # Notice that we are using >=, not strict >. That will include maxima that
    # are flat, that stretch few points. It will also include local minima,
    # but just as with local maxima that are too low below slope line, these
    # will be filtered out.
    is_maximum = np.logical_and(spectrum[ibegin+1:iend-1] >= spectrum[ibegin:iend-2],
                                spectrum[ibegin+1:iend-1] >= spectrum[ibegin+2:iend])

    # Get local maxima indices. (Note that where return tuple for each dimension).
    lmi = np.where(is_maximum)[0]

    # No local maxima, return.
    if len(lmi) == 0:
        return ind_fill

    # Make it relative to input array - spectrum.
    lmi += ibegin + 1

    # Get local maxima.
    local_maxima = spectrum[lmi]

    # Filter those maxima that cause slope between them to change direction.
    # This makes remaining maxima, satisfy quasy-convexity condition.
    slope_dir = spectrum[iend-1] - spectrum[ibegin]
    filtered_indices = []
    if slope_dir >= 0.0:
        last_included_value = spectrum[ibegin]
        for i in range(len(local_maxima)):
            lm = local_maxima[i]
            if lm > last_included_value:
                filtered_indices.append(lmi[i])
                last_included_value = lm
    else:
        # Slope is negative. Start from back.
        last_included_value = spectrum[iend-1]
        for i in range(len(local_maxima) - 1, -1, -1):
            lm = local_maxima[i]
            if lm > last_included_value:
                filtered_indices.append(lmi[i])
                last_included_value = lm
        filtered_indices.reverse()

    # Take only valid local maxima indices.
    lmi = filtered_indices

    # If there is no valid local maxima indices, return.
    if len(lmi) == 0:
        return ind_fill

    # Add indices to result, and process subregions between them with convex hull
    # algorithm, to make sure all input points and below resulting hull.
    next_ibegin = ibegin
    for i in lmi:
        # There should be at least 1 point between edges, to call _find_indices_in_range.
        # However, these are to local maxima, and if there is one point between them,
        # it must be below both. So only for two points inside region borders
        # call _find_indices_in_range.
        if i > next_ibegin + 2:
            # Put hull around points in subregion.
            ind_fill = _find_indices_in_range(
                spectrum, bands, False, indices, ind_fill, next_ibegin, i + 1)
        indices[ind_fill] = i
        ind_fill += 1
        next_ibegin = i

    # Don't miss the last range.
    ind_fill = _find_indices_in_range(
        spectrum, bands, False, indices, ind_fill, lmi[-1], iend)

    return ind_fill


def _find_indices_in_range(spectrum, bands, segmented, indices, ind_fill, ibegin, iend):
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
    imax = np.argmax(bands[ibegin:iendi] * naxis_x +
                     spectrum[ibegin:iendi] * naxis_y) + ibegin

    # If first point is maximum, then all others are "below" the axis,
    # which means this is concave region.
    if imax == ibegin:
        # If we are in segmented upper hull mode, then segment concave region.
        # For that to make sense, we need at least 3 elements between edges.
        if segmented and iend - ibegin > 5:
            ind_fill = _segment_concave_region(
                spectrum, bands, indices, ind_fill, ibegin, iend)
        return ind_fill

    # Repeat same procedure on the left side, if there are enough points left.
    # At least 1 is required between first and last point in range.
    if imax > ibegin + 1:
        ind_fill = _find_indices_in_range(
            spectrum, bands, segmented, indices, ind_fill, ibegin, imax + 1)

    # Push middle index.
    indices[ind_fill] = imax
    ind_fill += 1

    # Repeat for the right side.
    if imax < iend - 2:
        ind_fill = _find_indices_in_range(
            spectrum, bands, segmented, indices, ind_fill, imax, iend)

    return ind_fill


def _find_continuum_points_recursive(spectrum, bands, segmented, indices):
    n = len(spectrum)
    indices[0] = 0
    ind_fill = 1

    ind_fill = _find_indices_in_range(
        spectrum, bands, segmented, indices, ind_fill, 0, n)
    indices[ind_fill] = n - 1
    indices = indices[:ind_fill + 1]

    return (bands[indices], spectrum[indices])


def _process_continuum(spectra, bands, remove_continuum, segmented, out):
    if not isinstance(spectra, np.ndarray):
        raise TypeError('Expected spectra to be a numpy.ndarray.')
    if not isinstance(bands, np.ndarray):
        raise TypeError('Expected bands to be a numpy.ndarray.')
    if out is not None and not isinstance(out, np.ndarray):
        raise TypeError('Expected out to be a numpy.ndarray or None.')
    if len(spectra.shape) not in (1, 2, 3):
        raise ValueError('Expected spectra to be 1d, 2d, or 3d array.')
    if len(bands.shape) != 1:
        raise ValueError('Expected bands to be 1d array.')
    if out is not None and not np.array_equal(out.shape, spectra.shape):
        raise ValueError('Expected out to be same shape as spectra.')

    out = np.empty_like(spectra) if out is None else out

    # In case we remove continuum, always divide out by continuum,
    # to avoid creating additional temporary array.
    if spectra is not out and remove_continuum:
        out[:] = spectra[:]

    original_shape = spectra.shape
    nbands = original_shape[-1]

    interp = np.interp
    indices = np.empty(nbands, np.int64)

    if len(spectra.shape) == 1:
        points = _find_continuum_points_recursive(
            spectra, bands, segmented, indices)
        continuum = interp(bands, points[0], points[1])
        if remove_continuum:
            out /= continuum
        else:
            out[:] = continuum
    elif len(spectra.shape) == 2:
        for i in range(spectra.shape[0]):
            points = _find_continuum_points_recursive(
                spectra[i], bands, segmented, indices)
            continuum = interp(bands, points[0], points[1])
            if remove_continuum:
                out[i, :] /= continuum
            else:
                out[i, :] = continuum
    else:
        for i in range(spectra.shape[0]):
            for j in range(spectra.shape[1]):
                points = _find_continuum_points_recursive(
                    spectra[i, j], bands, segmented, indices)
                continuum = interp(bands, points[0], points[1])
                if remove_continuum:
                    out[i, j, :] /= continuum
                else:
                    out[i, j, :] = continuum

    return out


def continuum_points(spectrum, bands, mode='convex'):
    '''Returns points of spectra that belong to it's continuum.

    Arguments:

        `spectrum` (:class:`numpy.ndarray`)

            1d :class:`numpy.ndarray` holding spectral signature.

        `bands` (:class:`numpy.ndarray`):

            1d :class:`numpy.ndarray`, holding band values of spectra.
            Length of `bands` should be the same as `spectrum`.
            Note that bands should be sorted in ascending order (which is often
            not the case with AVIRIS), otherwise unexpected results could occur.

        `mode` (string, default 'convex'):

            Default mode is 'convex' which returns convex upper hull of the
            spectrum. Another supported mode is 'segmented' which builds
            segmented upper hull. This is useful to identify more detailed
            contour of the spectrum, but without strong absorption bands.

    Returns:

        2-tuple, with each element being :class:`numpy.ndarray`.
        First element contains reflectance values of points that belong to
        continuum. Second element contains corresponding bands.
        By applying linear interpolation to this data as x and y, we get
        continuum of spectrum. However this function is particularly useful to
        applying other interpolations or any other processing on these points.
    '''
    if not isinstance(spectrum, np.ndarray):
        raise TypeError('Expected spectra to be a numpy.ndarray.')
    if not isinstance(bands, np.ndarray):
        raise TypeError('Expected bands to be a numpy.ndarray.')
    if len(spectrum.shape) != 1:
        raise ValueError('Expected spectra to be 1d array.')
    if len(bands.shape) != 1:
        raise ValueError('Expected bands to be 1d array.')

    indices = np.empty_like(spectrum, dtype='int64')
    return _find_continuum_points_recursive(spectrum, bands, mode == 'segmented', indices)


def spectral_continuum(spectra, bands, mode='convex', out=None):
    '''Returns continua of spectra.
    Continuum is defined as convex hull of spectra.

    Arguments:

        `spectra` (:class:`numpy.ndarray`)

            Can be 1d, 2d or 3d :class:`numpy.ndarray`, where last dimension
            holds individual spectra.

        `bands` (:class:`numpy.ndarray`):

            1d :class:`numpy.ndarray`, holding band values of spectra.
            Length of `bands` should be the same as last dimension of `spectra`.
            Note that bands should be sorted in ascending order (which is often
            not the case with AVIRIS), otherwise unexpected results could occur.

        `mode` (string, default 'convex'):

            Default mode is 'convex' which returns convex upper hull of the
            spectrum. Another supported mode is 'segmented' which builds
            segmented upper hull. This is useful to identify more detailed
            contour of the spectrum, but without strong absorption bands.

        `out` (:class:`numpy.ndarray`, default None):

            If provided, it must have same type and same shape as `spectra`,
            and it will hold the result, and will be returned as result of this
            function.

    Returns:

        A :class:`numpy.ndarray` of continua for each spectrum in spectra.
        It same type and shape as spectra. If `out` is provided, `out` will be
        returned.
    '''
    return _process_continuum(spectra, bands, False, mode == 'segmented', out)


def remove_continuum(spectra, bands, mode='convex', out=None):
    '''Returns spectra with continuum removed.
    Continuum is defined as convex hull of spectra. Continuum is removed from
    spectra by dividing spectra by its continuum.

    Arguments:

        `spectra` (:class:`numpy.ndarray`)

            Can be 1d, 2d or 3d :class:`numpy.ndarray`, where last dimension
            holds individual spectra.

        `bands` (:class:`numpy.ndarray`):

            1d :class:`numpy.ndarray`, holding band values of spectra.
            Length of `bands` should be the same as last dimension of `spectra`.
            Note that bands should be sorted in ascending order (which is often
            not the case with AVIRIS), otherwise unexpected results could occur.

        `mode` (string, default 'convex'):

            Default mode is 'convex' which removes convex upper hull of the
            spectrum. Another supported mode is 'segmented' which removes
            segmented upper hull. This is useful to identify two or more small
            features instead of one large feature.

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
    return _process_continuum(spectra, bands, True, mode == 'segmented', out)
