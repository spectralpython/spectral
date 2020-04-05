'''
Runs unit tests of functions associated with the ECOSTRESS database.

To run the unit tests, type the following from the system command line:

    # python -m spectral.tests.ecostress

Note that the ECOSTRESS database must be requested so if the data files are
not located on the local file system, these tests will be skipped.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os
from numpy.testing import assert_almost_equal

import spectral as spy
from spectral.io.aviris import read_aviris_bands
from spectral.tests import testdir
from spectral.tests.spytest import SpyTest

ECOSTRESS_DATA_DIR = os.path.join(os.path.split(__file__)[0],
                                  'data/ecostress')
ECOSTRESS_DB = os.path.join(testdir, 'ecostress.db')
AVIRIS_BAND_FILE = os.path.join(os.path.split(__file__)[0], 'data/92AV3C.spc')

class ECOSTRESSDatabaseCreationTest(SpyTest):
    '''Tests that SpyFile memmap interfaces read and write properly.'''
    def __init__(self):
        pass

    def setup(self):
        if not os.path.isdir(testdir):
            os.makedirs(testdir)

    def test_create_database(self):
        '''Test creating new database from ECOSTRESS data files.'''
        db = spy.EcostressDatabase.create(ECOSTRESS_DB,
                                          ECOSTRESS_DATA_DIR)
        assert(list(db.query('SELECT COUNT() FROM Spectra'))[0][0] == 3)

class ECOSTRESSDatabaseTest(SpyTest):
    '''Tests that SpyFile memmap interfaces read and write properly.'''
    def __init__(self):
        pass

    def setup(self):
        self.db = spy.EcostressDatabase(ECOSTRESS_DB)

    def test_read_signatures(self):
        '''Cat get spectra from the opened database.'''
        assert(list(self.db.query('SELECT COUNT() FROM Spectra'))[0][0] == 3)

    def test_create_envi_lib(self):
        '''Can resample spectra and create an ENVI spectral library.'''
        bands = read_aviris_bands(AVIRIS_BAND_FILE)
        cursor = self.db.query('SELECT SpectrumID FROM Spectra')
        ids = [r[0] for r in cursor]
        bands.centers = [x / 1000. for x in bands.centers]
        bands.bandwidths = [x / 1000. for x in bands.bandwidths]
        slib = self.db.create_envi_spectral_library(ids, bands)
        assert(slib.spectra.shape == (3, 220))

def run():
    print('\n' + '-' * 72)
    print('Running ECOSTRESS tests.')
    print('-' * 72)
    ECOSTRESSDatabaseCreationTest().run()
    ECOSTRESSDatabaseTest().run()

if __name__ == '__main__':
    from spectral.tests.run import parse_args, reset_stats, print_summary
    parse_args()
    reset_stats()
    run()
    print_summary()
