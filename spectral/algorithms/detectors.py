#########################################################################
#
#   detectors.py - This file is part of the Spectral Python (SPy)
#   package.
#
#   Copyright (C) 2012-2013 Thomas Boggs
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
Spectral target detection algorithms
'''

__all__ = ['MatchedFilter', 'matched_filter', 'RX', 'rx']

import numpy as np
from spectral.algorithms.transforms import LinearTransform

class MatchedFilter(LinearTransform):
    r'''A callable linear matched filter.

    Given target/background means and a common covariance matrix, the matched
    filter response is given by:

    .. math::

        y=\frac{(\mu_t-\mu_b)^T\Sigma^{-1}(x-\mu_b)}{(\mu_t-\mu_b)^T\Sigma^{-1}(\mu_t-\mu_b)}

    where :math:`\mu_t` is the target mean, :math:`\mu_b` is the background
    mean, and :math:`\Sigma` is the covariance.
    '''

    def __init__(self, background, target):
        '''Creates the filter, given background/target means and covariance.

        Arguments:

            `background` (`GaussianStats`):

                The Gaussian statistics for the background (e.g., the result
                of calling :func:`calc_stats`).

            `target` (ndarray):

                Length-K target mean
        '''
        from math import sqrt
        from spectral.algorithms.transforms import LinearTransform

        self.background = background
        self.u_b = background.mean
        self.u_t = target
        self._whitening_transform = None

        d_tb = (target - self.u_b)
        self.d_tb = d_tb
        C_1 = background.inv_cov
        self.C_1 = C_1

        # Normalization coefficient (inverse of  squared Mahalanobis distance
        # between u_t and u_b)
        self.coef = 1.0 / d_tb.dot(C_1).dot(d_tb)

        LinearTransform.__init__(
            self, (self.coef * d_tb).dot(C_1), pre=-self.u_b)

    def whiten(self, X):
        '''Transforms data to the whitened space of the background.

        Arguments:

            `X` (ndarray):

                Size (M,N,K) or (M*N,K) array of length K vectors to transform.

        Returns an array of same size as `X` but linearly transformed to the
        whitened space of the filter.
        '''
        import math
        from spectral.algorithms.transforms import LinearTransform
        from spectral.algorithms.spymath import matrix_sqrt

        if self._whitening_transform is None:
            A = math.sqrt(self.coef) * self.background.sqrt_inv_cov
            self._whitening_transform = LinearTransform(A, pre=-self.u_b)
        return self._whitening_transform(X)

def matched_filter(X, target, background=None, window=None, cov=None):
    r'''Computes a linear matched filter target detector score.

    Usage:

        y = matched_filter(X, target, background)

        y = matched_filter(X, target, window=<win> [, cov=<cov>])
        
    Given target/background means and a common covariance matrix, the matched
    filter response is given by:

    .. math::

        y=\frac{(\mu_t-\mu_b)^T\Sigma^{-1}(x-\mu_b)}{(\mu_t-\mu_b)^T\Sigma^{-1}(\mu_t-\mu_b)}

    where :math:`\mu_t` is the target mean, :math:`\mu_b` is the background
    mean, and :math:`\Sigma` is the covariance.

    Arguments:

        `X` (numpy.ndarray):

            For the first calling method shown, `X` can be an image with
            shape (R, C, B) or an ndarray of shape (R * C, B). If the
            `background` keyword is given, it will be used for the image
            background statistics; otherwise, background statistics will be
            computed from `X`.

            If the `window` keyword is given, `X` must be a 3-dimensional
            array and background statistics will be computed for each point
            in the image using a local window defined by the keyword.

        `target` (ndarray):

            Length-K vector specifying the target to be detected.

        `background` (`GaussianStats`):

            The Gaussian statistics for the background (e.g., the result
            of calling :func:`calc_stats` for an image). This argument is not
            required if `window` is given.

        `window` (2-tuple of odd integers):

            Must have the form (`inner`, `outer`), where the two values
            specify the widths (in pixels) of inner and outer windows centered
            about the pixel being evaulated. Both values must be odd integers.
            The background mean and covariance will be estimated from pixels
            in the outer window, excluding pixels within the inner window. For
            example, if (`inner`, `outer`) = (5, 21), then the number of
            pixels used to estimate background statistics will be
            :math:`21^2 - 5^2 = 416`. If this argument is given, `background`
            is not required (and will be ignored, if given).

            The window is modified near image borders, where full, centered
            windows cannot be created. The outer window will be shifted, as
            needed, to ensure that the outer window still has height and width
            `outer` (in this situation, the pixel being evaluated will not be
            at the center of the outer window). The inner window will be
            clipped, as needed, near image borders. For example, assume an
            image with 145 rows and columns. If the window used is
            (5, 21), then for the image pixel at (0, 0) (upper left corner),
            the the inner window will cover `image[:3, :3]` and the outer
            window will cover `image[:21, :21]`. For the pixel at (50, 1), the
            inner window will cover `image[48:53, :4]` and the outer window
            will cover `image[40:51, :21]`.
            
        `cov` (ndarray):

            An optional covariance to use. If this parameter is given, `cov`
            will be used for all matched filter calculations (background
            covariance will not be recomputed in each window). Only the
            background mean will be recomputed in each window). If the
            `window` argument is specified, providing `cov` will allow the
            result to be computed *much* faster.

    Returns numpy.ndarray:

        The return value will be the matched filter scores distance) for each
        pixel given.  If `X` has shape (R, C, K), the returned ndarray will
        have shape (R, C).
    '''
    from exceptions import ValueError
    if background is not None and window is not None:
        raise ValueError('`background` and `window` are mutually ' \
                         'exclusive arguments.')
    if window is not None:
        from .spatial import map_outer_window_stats
        def mf_wrapper(bg, x):
            return MatchedFilter(bg, target)(x)
        return map_outer_window_stats(mf_wrapper, X, window[0], window[1],
                                      dim_out=1, cov=cov)
    else:
        from spectral.algorithms.algorithms import calc_stats
        if background is None:
            background = calc_stats(X)
        return MatchedFilter(background, target)(X)


class RX():
    r'''An implementation of the RX anomaly detector. Given the mean and
    covariance of the background, this detector returns the squared Mahalanobis
    distance of a spectrum according to

    .. math::

        y=(x-\mu_b)^T\Sigma^{-1}(x-\mu_b)

    where `x` is the unknown pixel spectrum, :math:`\mu_b` is the background
    mean, and :math:`\Sigma` is the background covariance.

    References:

    Reed, I.S. and Yu, X., "Adaptive multiple-band CFAR detection of an optical
    pattern with unknown spectral distribution," IEEE Trans. Acoust.,
    Speech, Signal Processing, vol. 38, pp. 1760-1770, Oct. 1990.
    '''
    dim_out=1

    def __init__(self, background=None):
        '''Creates the detector, given optional background/target stats.

        Arguments:

            `background` (`GaussianStats`, default None):

            The Gaussian statistics for the background (e.g., the result
            of calling :func:`calc_stats`). If no background stats are
            provided, they will be estimated based on data passed to the
            detector.
        '''
        from math import sqrt
        if background is not None:
            self.set_background(background)
        else:
            self.background = None

    def set_background(self, stats):
        '''Sets background statistics to be used when applying the detector.'''
        self.background = stats

    def __call__(self, X):
        '''Applies the RX anomaly detector to X.

        Arguments:

            `X` (numpy.ndarray):

                For an image with shape (R, C, B), `X` can be a vector of
                length B (single pixel) or an ndarray of shape (R, C, B) or
                (R * C, B).

        Returns numpy.ndarray or float:

            The return value will be the RX detector score (squared Mahalanobis
            distance) for each pixel given.  If `X` is a single pixel, a float
            will be returned; otherwise, the return value will be an ndarray
            of floats with one less dimension than the input.
        '''
        from spectral.algorithms.algorithms import calc_stats
        if not isinstance(X, np.ndarray):
            raise TypeError('Expected a numpy.ndarray.')

        if self.background is None:
            self.set_background(calc_stats(X))

        X = (X - self.background.mean)
        C_1 = self.background.inv_cov

        ndim = X.ndim
        shape = X.shape

        if ndim == 1:
            return X.dot(C_1).dot(X)

        if ndim == 3:
            X = X.reshape((-1, X.shape[-1]))

        A = X.dot(C_1)
        r = np.einsum('ij,ij->i', A, X)
        return r.reshape(shape[:-1])

        # I tried using einsum for the above calculations but, surprisingly,
        # it was *much* slower than using dot & sum. Need to figure out if
        # that is due to multithreading or some other reason.

#        print 'ndim =', ndim
#        if ndim == 1:
#            return np.einsum('i,ij,j', X, self.background.inv_cov, X)
#        if ndim == 3:
#            return np.einsum('ijk,km,ijm->ij',
#                             X, self.background.inv_cov, X).squeeze()
#        elif ndim == 2:
#            return np.einsum('ik,km,im->i',
#                             X, self.background.inv_cov, X).squeeze()
#        else:
#            raise Exception('Unexpected number of dimensions.')
#

def rx(X, background=None, window=None, cov=None):
    r'''Computes RX anomaly detector scores.

    Usage:

        y = rx(X [, background=bg])

        y = rx(X, window=(inner, outer) [, cov=C])

    The RX anomaly detector produces a detection statistic equal to the 
    squared Mahalanobis distance of a spectrum from a background distribution
    according to

    .. math::

        y=(x-\mu_b)^T\Sigma^{-1}(x-\mu_b)

    where `x` is the pixel spectrum, :math:`\mu_b` is the background
    mean, and :math:`\Sigma` is the background covariance.

    Arguments:

        `X` (numpy.ndarray):

            For the first calling method shown, `X` can be an image with
            shape (R, C, B) or an ndarray of shape (R * C, B). If the
            `background` keyword is given, it will be used for the image
            background statistics; otherwise, background statistics will be
            computed from `X`.

            If the `window` keyword is given, `X` must be a 3-dimensional
            array and background statistics will be computed for each point
            in the image using a local window defined by the keyword.


        `background` (`GaussianStats`):

            The Gaussian statistics for the background (e.g., the result
            of calling :func:`calc_stats`). If no background stats are
            provided, they will be estimated based on data passed to the
            detector.

        `window` (2-tuple of odd integers):

            Must have the form (`inner`, `outer`), where the two values
            specify the widths (in pixels) of inner and outer windows centered
            about the pixel being evaulated. Both values must be odd integers.
            The background mean and covariance will be estimated from pixels
            in the outer window, excluding pixels within the inner window. For
            example, if (`inner`, `outer`) = (5, 21), then the number of
            pixels used to estimate background statistics will be
            :math:`21^2 - 5^2 = 416`.

            The window are modified near image borders, where full, centered
            windows cannot be created. The outer window will be shifted, as
            needed, to ensure that the outer window still has height and width
            `outer` (in this situation, the pixel being evaluated will not be
            at the center of the outer window). The inner window will be
            clipped, as needed, near image borders. For example, assume an
            image with 145 rows and columns. If the window used is
            (5, 21), then for the image pixel at (0, 0) (upper left corner),
            the the inner window will cover `image[:3, :3]` and the outer
            window will cover `image[:21, :21]`. For the pixel at (50, 1), the
            inner window will cover `image[48:53, :4]` and the outer window
            will cover `image[40:51, :21]`.
            
        `cov` (ndarray):

            An optional covariance to use. If this parameter is given, `cov`
            will be used for all RX calculations (background covariance
            will not be recomputed in each window). Only the background
            mean will be recomputed in each window).

    Returns numpy.ndarray:

        The return value will be the RX detector score (squared Mahalanobis
        distance) for each pixel given.  If `X` has shape (R, C, B), the
        returned ndarray will have shape (R, C)..
    
    References:

    Reed, I.S. and Yu, X., "Adaptive multiple-band CFAR detection of an optical
    pattern with unknown spectral distribution," IEEE Trans. Acoust.,
    Speech, Signal Processing, vol. 38, pp. 1760-1770, Oct. 1990.
    '''
    from exceptions import ValueError
    if background is not None and window is not None:
        raise ValueError('`background` and `window` keywords are mutually ' \
                         'exclusive.')
    if window is not None:
        from .spatial import map_outer_window_stats
        rx = RX()
        def rx_wrapper(bg, x):
            rx.set_background(bg)
            return rx(x)
        return map_outer_window_stats(rx_wrapper, X, window[0], window[1],
                                      dim_out=1, cov=cov)
    else:
        return RX(background)(X)


