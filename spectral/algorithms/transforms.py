#########################################################################
#
#   transforms.py - This file is part of the Spectral Python (SPy)
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
Base classes for various types of transforms
'''

import numpy as np

class LinearTransform:
    def __init__(self, A, **kwargs):
        '''Arguments:
        
            `A` (class::`numpy.ndarrray`):
            
                An (J,K) array to be applied to length-K targets.
                    
        Keyword Argments:
        
            `pre` (scalar or length-K sequence):
            
                An additive offset to be applied prior to linear transformation.
                
            `post` (scalar or length-J sequence):
        
                An additive offset to be applied after linear transformation.
        '''
        
        self._pre = kwargs.get('pre', None)
        self._post = kwargs.get('post', None)
        if len(A.shape) == 1:
            self._A = A.reshape(((1,) + A.shape))
        else:
            self._A = A
    def __call__(self, X):
        '''Applies the linear transformation to the given data.
        
        Arguments:
        
            `X` (class::`numpy.ndarray):
            
                `X` is either an (M,N,K) array containing M*N length-K vectors
                to be transformed or it is an (R,K) array lf length-K vectors
                to be transformed.
                
        Returns an (M,N,J) or (R,J) array, depending on shape of `X`, where J
        is the length of the first dimension of the array `A` passed to
        __init__.
        '''
        shape = X.shape
        if len(shape) == 3:
            X = X.reshape((-1, shape[-1]))
            if self._pre != None:
                X = X + self._pre
            Y = np.dot(self._A, X.T).T
            if self._post != None:
                Y += self._post
            return Y.reshape((shape[:2] + (-1,))).squeeze()
        else:
            if self._pre != None:
                X = X + self._pre
            Y = np.dot(self._A, X.T).T
            if self._post != None:
                Y += self._post
            return Y     
