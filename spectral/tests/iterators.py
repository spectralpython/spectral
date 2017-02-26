#########################################################################
#
#   iterators.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2014 Thomas Boggs
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
'''Runs unit tests for iterators

To run the unit tests, type the following from the system command line:

    # python -m spectral.tests.iterators
'''

from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy.testing import assert_allclose
from .spytest import SpyTest
import spectral as spy

class IteratorTest(SpyTest):
    '''Tests various math functions.'''

    def setup(self):
        self.image = spy.open_image('92AV3C.lan')
        self.gt = spy.open_image('92AV3GT.GIS').read_band(0)

    def test_iterator_all(self):
        '''Iteration over all pixels.'''
        from spectral.algorithms.algorithms import iterator
        data = self.image.load()
        pixels = data.reshape((-1, data.shape[-1]))
        sum = np.sum(pixels, 0)
        itsum = np.sum(np.array([x for x in iterator(data)]), 0)
        assert_allclose(sum, itsum)

    def test_iterator_nonzero(self):
        '''Iteration over all non-background pixels.'''
        from spectral.algorithms.algorithms import iterator
        data = self.image.load()
        classes = self.gt.ravel()
        pixels = data.reshape((-1, data.shape[-1]))
        sum = np.sum(pixels[classes > 0], 0)
        itsum = np.sum(np.array([x for x in iterator(data, self.gt)]), 0)
        assert_allclose(sum, itsum)

    def test_iterator_index(self):
        '''Iteration over single ground truth index'''
        from spectral.algorithms.algorithms import iterator
        cls = 5
        data = self.image.load()
        classes = self.gt.ravel()
        pixels = data.reshape((-1, data.shape[-1]))
        sum = np.sum(pixels[classes == cls], 0)
        itsum = np.sum(np.array([x for x in iterator(data, self.gt, cls)]), 0)
        assert_allclose(sum, itsum)

    def test_iterator_ij_nonzero(self):
        '''Iteration over all non-background pixels.'''
        from spectral.algorithms.algorithms import iterator_ij
        data = self.image.load()
        classes = self.gt.ravel()
        pixels = data.reshape((-1, data.shape[-1]))
        sum = np.sum(pixels[classes > 0], 0)
        itsum = np.sum(np.array([data[ij] for ij in iterator_ij(self.gt)]), 0)
        assert_allclose(sum, itsum)

    def test_iterator_ij_index(self):
        '''Iteration over single ground truth index'''
        from spectral.algorithms.algorithms import iterator_ij
        cls = 5
        data = self.image.load()
        classes = self.gt.ravel()
        pixels = data.reshape((-1, data.shape[-1]))
        sum = np.sum(pixels[classes == cls], 0)
        itsum = np.sum(np.array([data[ij] for ij in iterator_ij(self.gt,
                                                                cls)]),
                       0)
        assert_allclose(sum, itsum)

    def test_iterator_spyfile(self):
        '''Iteration over SpyFile object for single ground truth index'''
        from spectral.algorithms.algorithms import iterator
        cls = 5
        data = self.image.load()
        classes = self.gt.ravel()
        pixels = data.reshape((-1, data.shape[-1]))
        sum = np.sum(pixels[classes == cls], 0)
        itsum = np.sum(np.array([x for x in iterator(self.image, self.gt, cls)]),
                                0)
        assert_allclose(sum, itsum)

    def test_iterator_spyfile_nomemmap(self):
        '''Iteration over SpyFile object without memmap'''
        from spectral.algorithms.algorithms import iterator
        cls = 5
        data = self.image.load()
        classes = self.gt.ravel()
        pixels = data.reshape((-1, data.shape[-1]))
        sum = np.sum(pixels[classes == cls], 0)
        image = spy.open_image('92AV3C.lan')
        itsum = np.sum(np.array([x for x in iterator(image, self.gt, cls)]), 0)
        assert_allclose(sum, itsum)


def run():
    print('\n' + '-' * 72)
    print('Running iterator tests.')
    print('-' * 72)
    test = IteratorTest()
    test.run()

if __name__ == '__main__':
    from spectral.tests.run import parse_args, reset_stats, print_summary
    parse_args()
    reset_stats()
    run()
    print_summary()
