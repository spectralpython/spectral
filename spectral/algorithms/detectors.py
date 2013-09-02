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

__all__ = ['MatchedFilter', 'RX', 'RXW', 'rx']

import numpy as np
from spectral.algorithms.transforms import LinearTransform as _LinearTransform


class MatchedFilter(_LinearTransform):
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
        C_1 = np.linalg.inv(background.cov)
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
            A = math.sqrt(self.coef) * matrix_sqrt(self.C_1, True)
            self._whitening_transform = LinearTransform(A, pre=-self.u_b)
        return self._whitening_transform(X)


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
        from spectral.algorithms.spymath import matrix_sqrt
        self.background = stats
        self.u_b = stats.mean
        # Matrix square root of inverse of cov: C**(-1/2)
        self.C_1_2 = matrix_sqrt(np.linalg.inv(stats.cov))

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

        X = (X - self.u_b)
        ndim = X.ndim
        shape = X.shape

        if ndim == 1:
            return sum(self.C_1_2.dot(X)**2)

        if ndim == 3:
            X = X.reshape((-1, X.shape[-1]))

        X = self.C_1_2.dot(X.T)
        r = np.sum(X * X, 0)
        return r.reshape(shape[:-1])

class RXW():
    r'''An RX anomaly detector using windowed background statistics. Given the
     mean and covariance of the background, this detector returns the squared
    Mahalanobis distance of a spectrum according to

    .. math::

        y=(x-\mu_b)^T\Sigma^{-1}(x-\mu_b)

    where `x` is the unknown pixel spectrum, :math:`\mu_b` is the background
    mean, and :math:`\Sigma` is the background covariance.

    References:

    Reed, I.S. and Yu, X., "Adaptive multiple-band CFAR detection of an optical
    pattern with unknown spectral distribution," IEEE Trans. Acoust.,
    Speech, Signal Processing, vol. 38, pp. 1760-1770, Oct. 1990.
    '''

    def __init__(self, window):
        '''Creates a detector with the given inner/outer window.

        Arguments:

            `window` (2-tuple of odd integers):

            `window` must have the form (`inner`, `outer`), where inner and
            outer are both odd-valued integers with `inner` < `outer`.
        '''
        from exceptions import ValueError
        (inner, outer) = window
        if inner % 2 == 0 or outer % 2 == 0:
            raise ValueError('Inner and outer window widths must be odd.')
        if inner >= outer:
            raise ValueError('Inner window must be smaller than outer.')

        self.window = window[:]

    def __call__(self, image):
        '''Applies the RX anomaly detector to X.

        Arguments:

            `image` (numpy.ndarray):

                An image with shape (R, C, B).

        Returns numpy.ndarray:

            The return value will be the RX detector score (squared Mahalanobis
            distance) for each pixel given in `image`.
        '''
        import spectral
        window = self.window
        (R, C, B) = image.shape
        (R_in, R_out) = window[:]
        (a, b) = [(x - 1) / 2 for x in window]

        x = np.ones((R, C), dtype=np.float32) * -1.0

        if R_out**2 - R_in**2 < B:
            raise ValueError('Window size provides too few samples for ' \
                             'image data dimensionality.')

        create_mask = window_mask_creator(image.shape, (window))
        interior_mask = create_mask(R / 2, C / 2)[1].ravel()
        interior_indices = np.argwhere(interior_mask == 0).squeeze()

        (i_interior_start, i_interior_stop) = (b, R - b)
        (j_interior_start, j_interior_stop) = (b, C - b)

        status = spectral._status
        status.display_percentage('Calculating RX scores: ')
        for i in range(C):
            for j in range(R):
                if i_interior_start <= i < i_interior_stop and \
                   j_interior_start <= j < j_interior_stop:
                    X = image[i - b : i + b + 1, j - b : j + b + 1, :]
                    indices = interior_indices
                else:
                    ((i0, i1, j0, j1), mask) = create_mask(i, j)
                    indices = np.argwhere(mask.ravel() == 0).squeeze()
                    X = image[i0 : i1, j0 : j1, :]
                X = np.take(X.reshape((-1, B)), indices, axis=0)
                m = np.mean(X, axis=0)
                Cov = np.cov(X, rowvar=False)
                r = image[i, j] - m
                x[i, j] = r.dot(np.linalg.inv(Cov)).dot(r)
            if i % (C / 10) == 0:
                status.update_percentage(100. * i / C)
        status.end_percentage()
        return x

def rx(X, **kwargs):
    r'''Computes RX anomaly detector scores.

    Usage:

        y = rx(X [, background=bg]

        y = rx(X, window=(inner, outer))

    The RX anomaly detector produces a detection statistic equal to the 
    mean and covariance of the background, this detector returns the squared
    Mahalanobis distance of a spectrum from a background distribution
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

    Keyword Arguments:

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
    if 'background' in kwargs  and 'window' in kwargs:
        raise ValueError('`background` and `window` keywords are mutually ' \
                         'exclusive.')
    if 'window' in kwargs:
        return RXW(kwargs['window'])(X)
    return RX(kwargs.get('background', None))(X)


def window_mask_creator(image_shape, window):
    '''Returns a function to give  inner/outer windows.

    Arguments:

        `image_shape` (tuple of integers):

            Specifies the dimensions of the image for which windows are to be
            produced. Only the first two dimensions (rows, columns) is used.

        `window` (2-tuple of integers):

            Specifies the sizes of the inner & outer windows. Both values
            must be odd integers.

    Return value:

        A function that accepts row and column indices as inputs and returns
        a 2-tuples with the following elements:

            1. The limits of the outer window:

                (row_start, row_stop, col_start, col_stop)

            2. A square ndarray that defines the inner/outer pixel mask for
               the window. The mask array contains zeros in the outer window
               and ones in the inner window.
    '''
    (R, C) = image_shape[:2]
    (R_in, R_out) = window
    assert(R_in % 2 + R_out % 2 == 2)
    (a, b) = [(x - 1) / 2 for x in window]
    def create_mask(i, j):
        istart_in = max(i - a, 0)
        istop_in = min(i + a + 1, R)
        jstart_in = max(j - a, 0)
        jstop_in = min(j + a + 1, C)
        istart_out = max(i - b, 0)
        istop_out = min(istart_out + R_out, R)
        if istop_out == R:
            istart_out = R - R_out
        jstart_out = max(j - b, 0)
        jstop_out = min(jstart_out + R_out, C)
        if jstop_out == C:
            jstart_out = C - R_out
        mask = np.zeros((R_out, R_out), dtype=np.int)
        mask[istart_in - istart_out : istop_in - istart_out,
             jstart_in - jstart_out : jstop_in - jstart_out] = 1
        return ((istart_out, istop_out, jstart_out, jstop_out), mask)
    return create_mask

