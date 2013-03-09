#########################################################################
#
#   math.py - This file is part of the Spectral Python (SPy)
#   package.
#
#   Copyright (C) 2001-2013 Thomas Boggs
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
Miscellaneous math functions
'''

import numpy as np


def matrix_sqrt(X, symmetric=False):
    '''Returns the matrix square root of X.

    Arguments:

        `X` (square class::`numpy.ndarrray`)

        `symmetric` (optional, False by default):

            If True, `X` is assumed to be symmetric, which speeds up
            calculation of the square root.

    Returns a class::`numpy.ndarray` `S`, such that S.dot(S) = X
    '''
    (vals, V) = np.linalg.eig(X)
    k = len(vals)
    SRV = np.zeros((k, k))
    SRV[np.diag_indices(k)] = np.sqrt(vals)
    if symmetric:
        return V.dot(SRV).dot(V.T)
    else:
        return V.dot(SRV).dot(np.linalg.inv(V))
