'''
Tests for iterators.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_iterators.py
'''

import numpy as np
from numpy.testing import assert_allclose

import spectral as spy
from spectral.algorithms.algorithms import iterator, iterator_ij
from spectral.tests.conftest import AV3C_LAN


class TestIterator:
    '''Tests various math functions.'''

    def test_iterator_all(self, av3c_image):
        '''Iteration over all pixels.'''
        data = av3c_image.load()
        pixels = data.reshape((-1, data.shape[-1]))
        sum = np.sum(pixels, 0)
        itsum = np.sum(np.array([x for x in iterator(data)]), 0)
        assert_allclose(sum, itsum)

    def test_iterator_nonzero(self, av3c_image, gt):
        '''Iteration over all non-background pixels.'''
        data = av3c_image.load()
        classes = gt.ravel()
        pixels = data.reshape((-1, data.shape[-1]))
        sum = np.sum(pixels[classes > 0], 0)
        itsum = np.sum(np.array([x for x in iterator(data, gt)]), 0)
        assert_allclose(sum, itsum)

    def test_iterator_index(self, av3c_image, gt):
        '''Iteration over single ground truth index'''
        cls = 5
        data = av3c_image.load()
        classes = gt.ravel()
        pixels = data.reshape((-1, data.shape[-1]))
        sum = np.sum(pixels[classes == cls], 0)
        itsum = np.sum(np.array([x for x in iterator(data, gt, cls)]), 0)
        assert_allclose(sum, itsum)

    def test_iterator_ij_nonzero(self, av3c_image, gt):
        '''Iteration over all non-background pixels.'''
        data = av3c_image.load()
        classes = gt.ravel()
        pixels = data.reshape((-1, data.shape[-1]))
        sum = np.sum(pixels[classes > 0], 0)
        itsum = np.sum(np.array([data[ij] for ij in iterator_ij(gt)]), 0)
        assert_allclose(sum, itsum)

    def test_iterator_ij_index(self, av3c_image, gt):
        '''Iteration over single ground truth index'''
        cls = 5
        data = av3c_image.load()
        classes = gt.ravel()
        pixels = data.reshape((-1, data.shape[-1]))
        sum = np.sum(pixels[classes == cls], 0)
        itsum = np.sum(np.array([data[ij] for ij in iterator_ij(gt,
                                                                cls)]),
                       0)
        assert_allclose(sum, itsum)

    def test_iterator_spyfile(self, av3c_image, gt):
        '''Iteration over SpyFile object for single ground truth index'''
        cls = 5
        data = av3c_image.load()
        classes = gt.ravel()
        pixels = data.reshape((-1, data.shape[-1]))
        sum = np.sum(pixels[classes == cls], 0)
        itsum = np.sum(np.array([x for x in iterator(av3c_image, gt, cls)]),
                       0)
        assert_allclose(sum, itsum)

    def test_iterator_spyfile_nomemmap(self, av3c_image, gt):
        '''Iteration over SpyFile object without memmap'''
        cls = 5
        data = av3c_image.load()
        classes = gt.ravel()
        pixels = data.reshape((-1, data.shape[-1]))
        sum = np.sum(pixels[classes == cls], 0)
        image = spy.open_image(AV3C_LAN)
        itsum = np.sum(np.array([x for x in iterator(image, gt, cls)]), 0)
        assert_allclose(sum, itsum)
