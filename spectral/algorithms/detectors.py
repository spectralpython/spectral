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

__all__ = ['MatchedFilter', 'matched_filter', 'RX', 'rx',
           'WindowedGaussianBackgroundMapper']

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

class MatchedFilterWrapper(object):
    '''Wrapper for using MatchedFilter with WindowedGaussianBackgroundMapper.
    '''
    def __init__(self, target=None, background=None):
        self.target = None
        self.background = None
        if target is not None:
            self.set_target(target)
        if background is not None:
            self.set_background(background)

    def set_target(self, target):
        self.target = target
        if self.target is not None and self.background is not None:
            self.mf = MatchedFilter(self.background, self.target)

    def set_background(self, background):
        self.background = background
        if self.target is not None and self.background is not None:
            self.mf = MatchedFilter(self.background, self.target)

    def __call__(self, X):
        return self.mf(X)

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
        mf = MatchedFilterWrapper(target, background)
        wmf = WindowedGaussianBackgroundMapper(window=window,
                                               function=mf,
                                               cov=cov,
                                               dim_out=1)
        return wmf(X)
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

class WindowedGaussianBackgroundMapper(object):
    '''
    '''
    def __init__(self, window, function=None, cov=None, dim_out=None):
        '''Creates a detector with the given inner/outer window.

        Arguments:

            `window` (2-tuple of odd integers):

                `window` must have the form (`inner`, `outer`), where inner
                and outer are both odd-valued integers with `inner` < `outer`.

            `function` (callable object):

                A callable object that will be applied to each pixel when the
                __call__ method is called for this object. `function` must
                have the following properties:

                    - A `__call__` method that accepts a pixel spectrum

                    - A `set_background` method that accepts a `GaussianStats`
                      object.

                    - An optional `dim_out` integer member that specifies the
                      dimensionality of callable objects output. If this
                      member does not exist and it has not been specified as
                      an argument to this objects constructor, `dim_out` will
                      will be assumed to be 1.                

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
        '''
        from exceptions import ValueError
        (inner, outer) = window
        if inner % 2 == 0 or outer % 2 == 0:
            raise ValueError('Inner and outer window widths must be odd.')
        if inner >= outer:
            raise ValueError('Inner window must be smaller than outer.')
        self.window = window[:]
        self.callable = function
        self.cov = cov
        self.dim_out = dim_out
        self.create_mask = None

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
        from spectral.algorithms.algorithms import GaussianStats
        window = self.window
        (R, C, B) = image.shape
        (R_in, R_out) = window[:]
        (a, b) = [(x - 1) / 2 for x in window]

        if self.dim_out is not None:
            dim_out = self.dim_out
        elif hasattr(self.callable, 'dim_out') and \
          self.callable.dim_out is not None:
            dim_out = self.callable.dim_out
        else:
            dim_out = 1

        if dim_out > 1:
            x = np.ones((R, C, dim_out), dtype=np.float32) * -1.0
        else:
            x = np.ones((R, C), dtype=np.float32) * -1.0

        if self.cov is None and R_out**2 - R_in**2 < B:
            raise ValueError('Window size provides too few samples for ' \
                             'image data dimensionality.')

        if self.create_mask is not None:
            create_mask = self.create_mask
        else:
            create_mask = window_mask_creator(image.shape, window)

        interior_mask = create_mask(R / 2, C / 2, True)[2].ravel()
        interior_indices = np.argwhere(interior_mask == 0).squeeze()

        (i_interior_start, i_interior_stop) = (b, R - b)
        (j_interior_start, j_interior_stop) = (b, C - b)

        status = spectral._status
        status.display_percentage('Processing image: ')
        if self.cov is not None:
            # Since we already have the covariance, just use np.mean to get
            # means of the inner window and outer (including the inner), then
            # use those to calculate the mean of the outer window alone.
            background = GaussianStats(cov=self.cov)
            for i in range(R):
                for j in range(C):
                    (inner, outer) = create_mask(i, j, False)
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
                    self.callable.set_background(background)
                    x[i, j] = self.callable(image[i, j])
                if i % (R / 10) == 0:
                    status.update_percentage(100. * i / R)
        else:
            # Need to calculate both the mean and covariance for the outer
            # window (without the inner).
            for i in range(R):
                for j in range(C):
                    if i_interior_start <= i < i_interior_stop and \
                       j_interior_start <= j < j_interior_stop:
                        X = image[i - b : i + b + 1, j - b : j + b + 1, :]
                        indices = interior_indices
                    else:
                        (inner, (i0, i1, j0, j1), mask) = create_mask(i, j, True)
                        indices = np.argwhere(mask.ravel() == 0).squeeze()
                        X = image[i0 : i1, j0 : j1, :]
                    X = np.take(X.reshape((-1, B)), indices, axis=0)
                    mean = np.mean(X, axis=0)
                    cov = np.cov(X, rowvar=False)
                    background = GaussianStats(mean, cov)
                    self.callable.set_background(background)
                    x[i, j] = self.callable(image[i, j])
                if i % (R / 10) == 0:
                    status.update_percentage(100. * i / R)

        status.end_percentage()
        return x

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
        rx = RX()
        wrx = WindowedGaussianBackgroundMapper(window=window,
                                               function=rx,
                                               cov=cov,
                                               dim_out=1)
        return wrx(X)
    else:
        return RX(background)(X)


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

