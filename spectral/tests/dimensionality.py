#########################################################################
#
#   test.py - This file is part of the Spectral Python (SPy) package.
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
'''Runs unit tests for XXX

To run the unit tests, type the following from the system command line:

    # python -m spectral.tests.XXX
'''

from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy.testing import assert_allclose
import spectral as spy
from .spytest import SpyTest, test_method


class DimensionalityTest(SpyTest):
    '''Tests various math functions.'''

    def setup(self):
        self.data = spy.open_image('92AV3C.lan').load()

    def test_mnf_all_equals_data(self):
        '''Test that MNF transform with all components equals original data.'''
        data = self.data
        signal = spy.calc_stats(data)
        noise = spy.noise_from_diffs(data[117: 137, 85: 122, :])
        mnfr = spy.mnf(signal, noise)
        denoised = mnfr.denoise(data, num=data.shape[-1])
        assert(np.allclose(denoised, data))

    def test_ppi(self):
        '''Tests that ppi function runs'''
        data = self.data
        p = spy.ppi(data, 4)

    def test_ppi_threshold(self):
        '''Tests that ppi function runs with threshold arg'''
        data = self.data
        p = spy.ppi(data, 4, 10)

    def test_ppi_continues(self):
        '''Tests that running ppi with initial indices works as expected.'''
        data = self.data
        s = np.random.get_state()
        p = spy.ppi(data, 4)
        np.random.set_state(s)
        p2 = spy.ppi(data, 2)
        p2 = spy.ppi(data, 2, start=p2)
        assert(np.all(p == p2))

    def test_ppi_centered(self):
        '''Tests that ppi with mean-subtracted data works as expected.'''
        data = self.data
        s = np.random.get_state()
        p = spy.ppi(data, 4)
        
        np.random.set_state(s)
        data_centered = data - spy.calc_stats(data).mean
        p2 = spy.ppi(data_centered, 4)
        assert(np.all(p == p2))

    def test_pca_runs(self):
        '''Should be able to compute PCs and transform data.'''
        data = self.data
        xdata = spy.principal_components(data).transform(data)

    def test_pca_runs_from_stats(self):
        '''Should be able to pass image stats to PCA function.'''
        data = self.data
        stats = spy.calc_stats(data)
        xdata = spy.principal_components(stats).transform(data)

def run():
    print('\n' + '-' * 72)
    print('Running dimensionality tests.')
    print('-' * 72)
    test = DimensionalityTest()
    test.run()

if __name__ == '__main__':
    from spectral.tests.run import parse_args, reset_stats, print_summary
    parse_args()
    reset_stats()
    run()
    print_summary()
