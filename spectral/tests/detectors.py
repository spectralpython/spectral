#########################################################################
#
#   detectors.py - This file is part of the Spectral Python (SPy) package.
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
'''Runs unit tests for various target detectors

To run the unit tests, type the following from the system command line:

    # python -m spectral.tests.detectors
'''

import numpy as np
from numpy.testing import assert_allclose
from spytest import SpyTest, test_method
import spectral as spy


class MatchedFilterTest(SpyTest):
    def setup(self):
        from spectral.algorithms.detectors import MatchedFilter
        self.data = spy.open_image('92AV3C.lan').load()
        self.background = spy.calc_stats(self.data)
        self.target_ij = [33, 87]
#        self.target = self.data[33, 87]
        (i, j) = self.target_ij
        self.mf = MatchedFilter(self.background, self.data[i, j])

    @test_method
    def test_mf_bg_eq_zero(self):
        '''Matched Filter response of background should be zero.'''
        (i, j) = self.target_ij
        np.testing.assert_approx_equal(self.mf(self.background.mean), 0)
        
    @test_method
    def test_mf_target_eq_one(self):
        '''Matched Filter response of target should be one.'''
        (i, j) = self.target_ij
        np.testing.assert_approx_equal(self.mf(self.data[i, j]), 1)

    @test_method
    def test_mf_target_pixel_eq_one(self):
        '''Matched Filter response of target pixel should be one.'''
        (i, j) = self.target_ij
        np.testing.assert_approx_equal(self.mf(self.data)[i, j], 1)

    def run(self):
        '''Executes the test case.'''
        self.setup()
        self.test_mf_bg_eq_zero()
        self.test_mf_target_eq_one()
        self.test_mf_target_pixel_eq_one()
        self.finish()


class RXTest(SpyTest):
    def setup(self):
        from spectral.algorithms.detectors import rx, RX, RXW
        self.data = spy.open_image('92AV3C.lan').load()
        self.background = spy.calc_stats(self.data)

    @test_method
    def test_rx_bg_eq_zero(self):
        from spectral.algorithms.detectors import rx, RX, RXW
        d = rx(self.data)
        stats = spy.calc_stats(self.data)
        np.testing.assert_approx_equal(rx(stats.mean, background=stats), 0)
        
    @test_method
    def test_mf_target_eq_one(self):
        np.testing.assert_approx_equal(self.mf(self.target), 1)

    def run(self):
        '''Executes the test case.'''
        self.setup()
        self.test_rx_bg_eq_zero()
        self.finish()



def run():
    print '\n' + '-' * 72
    print 'Running target detector tests.'
    print '-' * 72
    for T in [MatchedFilterTest, RXTest]:
        T().run()

if __name__ == '__main__':
    from spectral.tests.run import parse_args, reset_stats, print_summary
    parse_args()
    reset_stats()
    run()
    print_summary()
