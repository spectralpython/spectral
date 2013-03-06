#########################################################################
#
#   detectors.py - This file is part of the Spectral Python (SPy)
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
Spectral target detection algorithms
'''

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

        LinearTransform.__init__(self, (self.coef * d_tb).dot(C_1), pre=-self.u_b)
    
    def whiten(self, X):
        '''Transforms data to the whitened space of the background.
        
        Arguments:
        
            `X` (ndarray):
            
                Size (M,N,K) or (M*N,K) array of length K vectors to transform.
                
        Returns an array of same size as `X` but linearly transformed to the
        whitened space of the filter.
        '''
        import math
        from spectral.algorithms.spymath import matrix_sqrt
        if self._whitening_transform == None:
            A = math.sqrt(self.coef) * matrix_sqrt(self.C_1, True)
            self._whitening_transform = LinearTransform(A, pre=-self.u_b)
        return self._whitening_transform(X)
