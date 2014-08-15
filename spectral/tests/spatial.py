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
'''Runs unit tests for various SPy spatial functions.

To run the unit tests, type the following from the system command line:

    # python -m spectral.tests.spatial
'''

from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy.testing import assert_allclose
from .spytest import SpyTest


class SpatialWindowTest(SpyTest):
    '''Tests various spatial functions.'''

    def setup(self):
        import spectral as spy
        self.data = spy.open_image('92AV3C.lan').load()

    def test_get_window_bounds(self):
        from spectral.algorithms.spatial import get_window_bounds
        assert(get_window_bounds(90, 90, 3, 7, 30, 40) == (29, 32, 37, 44))

    def test_get_window_bounds_border(self):
        from spectral.algorithms.spatial import get_window_bounds
        assert(get_window_bounds(90, 90, 3, 7, 0, 2) == (0, 3, 0, 7))

    def test_get_window_bounds_clipped(self):
        from spectral.algorithms.spatial import get_window_bounds_clipped
        assert(get_window_bounds_clipped(90, 90, 3, 7, 30, 40) \
               == (29, 32, 37, 44))

    def test_get_window_bounds_clipped_border(self):
        from spectral.algorithms.spatial import get_window_bounds_clipped
        assert(get_window_bounds_clipped(90, 90, 3, 7, 0, 2) == (0, 2, 0, 6))

    def test_map_window(self):
        '''Test computing spectra average over local window.'''
        from spectral.algorithms.spatial import map_window
        f = lambda X, ij: np.mean(X.reshape((-1, X.shape[-1])), axis=0)
        X = self.data
        y = map_window(f, X, (3, 5), (10, 50), (20, 40))
        t = np.mean(X[9:12, 18:23].reshape((-1, X.shape[-1])), axis=0)
        assert_allclose(y[0, 0], t)

    def test_map_window_clipped(self):
        '''Test spatial averaging near border with clipped window.'''
        from spectral.algorithms.spatial import map_window
        f = lambda X, ij: np.mean(X.reshape((-1, X.shape[-1])), axis=0)
        X = self.data
        y = map_window(f, X, (3, 5), (100, None), (100, None), border='clip')
        t = np.mean(X[-2:, -3:].reshape((-1, X.shape[-1])), axis=0)
        assert_allclose(y[-1, -1], t)

    def test_map_window_shifted(self):
        '''Test spatial averaging near border with shifted window.'''
        from spectral.algorithms.spatial import map_window
        f = lambda X, ij: np.mean(X.reshape((-1, X.shape[-1])), axis=0)
        X = self.data
        y = map_window(f, X, (3, 5), (100, None), (100, None), border='shift')
        t = np.mean(X[-3:, -5:].reshape((-1, X.shape[-1])), axis=0)
        assert_allclose(y[-1, -1], t)

    def test_map_window_stepped(self):
        '''Test spatial averaging with non-unity row/column step sizes.'''
        from spectral.algorithms.spatial import map_window
        f = lambda X, ij: np.mean(X.reshape((-1, X.shape[-1])), axis=0)
        X = self.data
        y = map_window(f, X, (3, 5), (30, 60, 3), (70, 100, 4), border='shift')
        t = np.mean(X[32:35, 72:77].reshape((-1, X.shape[-1])), axis=0)
        assert_allclose(y[1, 1], t)

def run():
    print('\n' + '-' * 72)
    print('Running spatial tests.')
    print('-' * 72)
    for T in [SpatialWindowTest]:
        T().run()

if __name__ == '__main__':
    from spectral.tests.run import parse_args, reset_stats, print_summary
    parse_args()
    reset_stats()
    run()
    print_summary()
