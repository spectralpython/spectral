#########################################################################
#
#   spymath.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2013 Thomas Boggs
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
'''Runs unit tests for various SPy math functions.

To run the unit tests, type the following from the system command line:

    # python -m spectral.tests.spymath
'''

import numpy as np
from numpy.testing import assert_allclose
from spytest import SpyTest, test_method


class SpyMathTest(SpyTest):
    '''Tests various math functions.'''

    def setup(self):
        import spectral as spy
        data = spy.open_image('92AV3C.lan').open_memmap()
        self.C = spy.calc_stats(data).cov
        self.X = np.random.rand(100, 100)

    @test_method
    def test_matrix_sqrt(self):
        from spectral.algorithms.spymath import matrix_sqrt
        S = matrix_sqrt(self.X)
        assert_allclose(S.dot(S), self.X)

    @test_method
    def test_matrix_sqrt_inv(self):
        from spectral.algorithms.spymath import matrix_sqrt
        S = matrix_sqrt(self.X, inverse=True)
        assert_allclose(S.dot(S), np.linalg.inv(self.X))

    @test_method
    def test_matrix_sqrt_sym(self):
        from spectral.algorithms.spymath import matrix_sqrt
        S = matrix_sqrt(self.C, symmetric=True)
        assert_allclose(S.dot(S), self.C, atol=1e-8)

    @test_method
    def test_matrix_sqrt_sym_inv(self):
        from spectral.algorithms.spymath import matrix_sqrt
        S = matrix_sqrt(self.C, symmetric=True, inverse=True)
        assert_allclose(S.dot(S), np.linalg.inv(self.C), atol=1e-8)

    def run(self):
        '''Executes the test case.'''
        self.setup()
        self.test_matrix_sqrt()
        self.test_matrix_sqrt_inv()
        self.test_matrix_sqrt_sym()
        self.test_matrix_sqrt_sym_inv()
        self.finish()


def run():
    print '\n' + '-' * 72
    print 'Running math tests.'
    print '-' * 72
    test = SpyMathTest()
    test.run()

if __name__ == '__main__':
    from spectral.tests.run import parse_args, reset_stats, print_summary
    parse_args()
    reset_stats()
    run()
    print_summary()
