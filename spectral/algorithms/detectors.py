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

__all__ = ['MatchedFilter', 'RX']

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
