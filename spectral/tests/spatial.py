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

class MapClassesTest(SpyTest):
    '''Test mapping of class indices between classification images.'''

    def setup(self):
        import spectral as spy
        self.gt = spy.open_image('92AV3GT.GIS').read_band(0)

    def test_map_class_ids_identity(self):
        '''Mapping a class image back to itself should yield identity map.'''
        from spectral.algorithms.spatial import map_class_ids
        gt = np.array(self.gt)
        d = map_class_ids(gt, gt)
        for i in set(gt.ravel()):
            assert(i in d)
        for (i, j) in d.items():
            assert(j == i)

    def test_map_class_ids_identity_unlabeled(self):
        '''Mapping a class image back to itself with an unlabeled class.'''
        from spectral.algorithms.spatial import map_class_ids
        gt = np.array(self.gt)
        d = map_class_ids(gt, gt, unlabeled=0)
        for i in set(gt.ravel()):
            assert(i in d)
        for (i, j) in d.items():
            assert(j == i)

    def test_map_class_ids_identity_multiple_unlabeled(self):
        '''Mapping a class image back to itself with unlabeled classes.'''
        from spectral.algorithms.spatial import map_class_ids
        gt = np.array(self.gt)
        d = map_class_ids(gt, gt, unlabeled=[2, 4])
        for i in set(gt.ravel()):
            assert(i in d)
        for (i, j) in d.items():
            assert(j == i)

    def test_map_class_ids_isomorphic(self):
        '''Test map_class_ids with isomorphic classes.'''
        from spectral.algorithms.spatial import map_class_ids
        gt = np.array(self.gt)
        gt2 = gt + 1
        d = map_class_ids(gt, gt2)
        for (i, j) in d.items():
            assert(j == i + 1)

    def test_map_class_ids_isomorphic_background(self):
        '''Test map_class_ids with isomorphic classes and background arg.'''
        from spectral.algorithms.spatial import map_class_ids
        gt = np.array(self.gt)
        gt2 = gt + 1
        d = map_class_ids(gt, gt2, unlabeled=0)
        assert(d[0] == 0)
        d.pop(0)
        for (i, j) in d.items():
            assert(j == i + 1)

    def test_map_class_ids_src_gt_dest(self):
        '''Test map_class_ids with more classes in source image.'''
        from spectral.algorithms.spatial import map_class_ids
        gt = np.array(self.gt)

        (i, j) = (100, 30)
        old_label = gt[i, j]
        new_label = max(set(gt.ravel())) + 10
        gt2 = np.array(gt)
        gt2[i, j] = new_label
        
        d = map_class_ids(gt2, gt)
        # There are enough pixels for each class that a new single-pixel class
        # should not be mapped to one of the existing classes.
        assert(d[new_label] not in gt)
        d.pop(new_label)
        for (i, j) in d.items():
            assert(j == i)

    def test_map_class_ids_dest_gt_src(self):
        '''Test map_class_ids with more classes in dest image.'''
        from spectral.algorithms.spatial import map_class_ids
        gt = np.array(self.gt)

        (i, j) = (100, 30)
        old_label = gt[i, j]
        new_label = max(set(gt.ravel())) + 10
        gt2 = np.array(gt)
        gt2[i, j] = new_label
        
        d = map_class_ids(gt, gt2)
        for (i, j) in d.items():
            assert(j == i)

    def test_map_classes_isomorphic(self):
        '''map_classes should map isomorphic class image back to original.'''
        from spectral.algorithms.spatial import map_class_ids, map_classes
        gt = np.array(self.gt)
        gt2 = gt + 1
        d = map_class_ids(gt2, gt)
        result = map_classes(gt2, d)
        assert(np.alltrue(result == gt))
        
    def test_map_fails_allow_unmapped_false(self):
        '''map_classes should raise ValueError if image has unmapped value.'''
        from spectral.algorithms.spatial import map_class_ids, map_classes
        from warnings import warn
        gt = np.array(self.gt)
        gt2 = gt + 1
        d = map_class_ids(gt2, gt)
        d.pop(1)
        try:
            result = map_classes(gt2, d)
        except ValueError:
            pass
        else:
            assert(False)
        
    def test_map_allow_unmapped_true(self):
        '''map_classes should raise ValueError if image has unmapped value.'''
        from spectral.algorithms.spatial import map_class_ids, map_classes
        from warnings import warn
        gt = np.array(self.gt)
        gt2 = gt + 1
        d = map_class_ids(gt2, gt)
        d.pop(1)
        result = map_classes(gt2, d, allow_unmapped=True)
        assert(np.alltrue(result[gt2 == 1] == 1))
        
def run():
    print('\n' + '-' * 72)
    print('Running spatial tests.')
    print('-' * 72)
    for T in [SpatialWindowTest, MapClassesTest]:
        T().run()

if __name__ == '__main__':
    from spectral.tests.run import parse_args, reset_stats, print_summary
    parse_args()
    reset_stats()
    run()
    print_summary()
