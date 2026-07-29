'''
Tests for various SPy spatial functions.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_spatial.py
'''

import numpy as np
from numpy.testing import assert_allclose
import pytest

from spectral.algorithms.spatial import (get_window_bounds,
                                         get_window_bounds_clipped,
                                         map_class_ids, map_classes,
                                         map_window)


@pytest.fixture
def av3c_data(av3c_image):
    return av3c_image.load()


class TestSpatialWindow:
    '''Tests various spatial functions.'''

    def test_get_window_bounds(self):
        assert (get_window_bounds(90, 90, 3, 7, 30, 40) == (29, 32, 37, 44))

    def test_get_window_bounds_border(self):
        assert (get_window_bounds(90, 90, 3, 7, 0, 2) == (0, 3, 0, 7))

    def test_get_window_bounds_clipped(self):
        assert (get_window_bounds_clipped(90, 90, 3, 7, 30, 40)
                == (29, 32, 37, 44))

    def test_get_window_bounds_clipped_border(self):
        assert (get_window_bounds_clipped(90, 90, 3, 7, 0, 2) == (0, 2, 0, 6))

    def test_map_window(self, av3c_data):
        '''Test computing spectra average over local window.'''
        def f(X, ij): return np.mean(X.reshape((-1, X.shape[-1])), axis=0)
        X = av3c_data
        y = map_window(f, X, (3, 5), (10, 50), (20, 40))
        t = np.mean(X[9:12, 18:23].reshape((-1, X.shape[-1])), axis=0)
        assert_allclose(y[0, 0], t)

    def test_map_window_clipped(self, av3c_data):
        '''Test spatial averaging near border with clipped window.'''
        def f(X, ij): return np.mean(X.reshape((-1, X.shape[-1])), axis=0)
        X = av3c_data
        y = map_window(f, X, (3, 5), (100, None), (100, None), border='clip')
        t = np.mean(X[-2:, -3:].reshape((-1, X.shape[-1])), axis=0)
        assert_allclose(y[-1, -1], t)

    def test_map_window_shifted(self, av3c_data):
        '''Test spatial averaging near border with shifted window.'''
        def f(X, ij): return np.mean(X.reshape((-1, X.shape[-1])), axis=0)
        X = av3c_data
        y = map_window(f, X, (3, 5), (100, None), (100, None), border='shift')
        t = np.mean(X[-3:, -5:].reshape((-1, X.shape[-1])), axis=0)
        assert_allclose(y[-1, -1], t)

    def test_map_window_stepped(self, av3c_data):
        '''Test spatial averaging with non-unity row/column step sizes.'''
        def f(X, ij): return np.mean(X.reshape((-1, X.shape[-1])), axis=0)
        X = av3c_data
        y = map_window(f, X, (3, 5), (30, 60, 3), (70, 100, 4), border='shift')
        t = np.mean(X[32:35, 72:77].reshape((-1, X.shape[-1])), axis=0)
        assert_allclose(y[1, 1], t)


class TestMapClasses:
    '''Test mapping of class indices between classification images.'''

    def test_map_class_ids_identity(self, gt):
        '''Mapping a class image back to itself should yield identity map.'''
        d = map_class_ids(gt, gt)
        for i in set(gt.ravel()):
            assert (i in d)
        for (i, j) in d.items():
            assert (j == i)

    def test_map_class_ids_identity_unlabeled(self, gt):
        '''Mapping a class image back to itself with an unlabeled class.'''
        d = map_class_ids(gt, gt, unlabeled=0)
        for i in set(gt.ravel()):
            assert (i in d)
        for (i, j) in d.items():
            assert (j == i)

    def test_map_class_ids_identity_multiple_unlabeled(self, gt):
        '''Mapping a class image back to itself with unlabeled classes.'''
        d = map_class_ids(gt, gt, unlabeled=[2, 4])
        for i in set(gt.ravel()):
            assert (i in d)
        for (i, j) in d.items():
            assert (j == i)

    def test_map_class_ids_isomorphic(self, gt):
        '''Test map_class_ids with isomorphic classes.'''
        gt2 = gt + 1
        d = map_class_ids(gt, gt2)
        for (i, j) in d.items():
            assert (j == i + 1)

    def test_map_class_ids_isomorphic_background(self, gt):
        '''Test map_class_ids with isomorphic classes and background arg.'''
        gt2 = gt + 1
        d = map_class_ids(gt, gt2, unlabeled=0)
        assert (d[0] == 0)
        d.pop(0)
        for (i, j) in d.items():
            assert (j == i + 1)

    def test_map_class_ids_src_gt_dest(self, gt):
        '''Test map_class_ids with more classes in source image.'''
        (i, j) = (100, 30)
        new_label = max(set(gt.ravel())) + 10
        gt2 = np.array(gt)
        gt2[i, j] = new_label

        d = map_class_ids(gt2, gt)
        # There are enough pixels for each class that a new single-pixel class
        # should not be mapped to one of the existing classes.
        assert (d[new_label] not in gt)
        d.pop(new_label)
        for (i, j) in d.items():
            assert (j == i)

    def test_map_class_ids_dest_gt_src(self, gt):
        '''Test map_class_ids with more classes in dest image.'''
        (i, j) = (100, 30)
        new_label = max(set(gt.ravel())) + 10
        gt2 = np.array(gt)
        gt2[i, j] = new_label

        d = map_class_ids(gt, gt2)
        for (i, j) in d.items():
            assert (j == i)

    def test_map_classes_isomorphic(self, gt):
        '''map_classes should map isomorphic class image back to original.'''
        gt2 = gt + 1
        d = map_class_ids(gt2, gt)
        result = map_classes(gt2, d)
        assert (np.all(result == gt))

    def test_map_fails_allow_unmapped_false(self, gt):
        '''map_classes should raise ValueError if image has unmapped value.'''
        gt2 = gt + 1
        d = map_class_ids(gt2, gt)
        d.pop(1)
        with pytest.raises(ValueError):
            map_classes(gt2, d)

    def test_map_allow_unmapped_true(self, gt):
        '''map_classes should raise ValueError if image has unmapped value.'''
        gt2 = gt + 1
        d = map_class_ids(gt2, gt)
        d.pop(1)
        result = map_classes(gt2, d, allow_unmapped=True)
        assert (np.all(result[gt2 == 1] == 1))
