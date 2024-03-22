'''
Miscellaneous math functions.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


def matrix_sqrt(X=None, symmetric=False, inverse=False, eigs=None):
    '''Returns the matrix square root of X.

    Arguments:

        `X` (square class::`numpy.ndarrray`)

        `symmetric` (bool, default False):

            If True, `X` is assumed to be symmetric, which speeds up
            calculation of the square root.

        `inverse` (bool, default False):

            If True, computes the matrix square root of inv(X).

        `eigs` (2-tuple):

            `eigs` must be a 2-tuple whose first element is an array of
            eigenvalues and whose second element is an ndarray of eigenvectors
            (individual eigenvectors are in columns). If this argument is
            provided, computation of the matrix square root is much faster. If
            this argument is provided, the `X` argument is ignored (in this
            case, it can be set to None).

    Returns a class::`numpy.ndarray` `S`, such that S.dot(S) = X
    '''
    if eigs is not None:
        (vals, V) = eigs
    else:
        (vals, V) = np.linalg.eig(X)
    if inverse is False:
        SRV = np.diag(np.sqrt(vals))
    else:
        SRV = np.diag(1. / np.sqrt(vals))
    if symmetric:
        return V.dot(SRV).dot(V.T)
    else:
        return V.dot(SRV).dot(np.linalg.inv(V))


def get_histogram_cdf_points(data, cdf_vals, ignore=None, mask=None):
    '''Returns input values corresponding to the data's CDF values.

    Arguments:

        `data` (ndarray):

            The data for which to determine the CDF values

        `cdf_vals` (sequence of floats):

            A sequence defining the CDF values for which the values of `data`
            should be returned. Each value should be in the range [0, 1]. For
            example, to get the values of `data` corresponding to the 1% lower
            tail and 5% upper tail, this argument would be (0.01, 0.95).

        `ignore` (numeric, default `None`):

            A scalar value that should be ignored when computing histogram
            points (e.g., a value that indicates bad data). If this value is
            not specified, all data are used.

    Return value:

        A list specifying the values in `data` that correspond to the
        associated CDF values in `cdf_vals`.
    '''
    data = data.ravel()
    if mask is not None:
        data = data[mask.ravel() != 0]
        if len(data) == 0:
            raise Exception('All pixels are masked.')
    if ignore is not None and ignore in data:
        data = data[np.where(data != ignore)]
        if len(data) == 0:
            raise Exception('No data to display after masking and ignoring.')
    isort = np.argsort(data)
    N = len(data)
    return [data[isort[int(x * (N - 1))]] for x in cdf_vals]
