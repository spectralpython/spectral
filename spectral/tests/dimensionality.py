'''
Runs unit tests for dimensionality reduction algorithms.

To run the unit tests, type the following from the system command line:

    # python -m spectral.tests.dimensionality
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy.testing import assert_allclose

import spectral as spy
from spectral.tests.spytest import SpyTest, test_method


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
