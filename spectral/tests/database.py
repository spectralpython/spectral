'''
Runs unit tests of functions associated with spectral databases.

To run the unit tests, type the following from the system command line:

    # python -m spectral.tests.database

Note that the ECOSTRESS database must be requested so if the data files are
not located on the local file system, these tests will be skipped.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import os
from numpy.testing import assert_almost_equal

import spectral as spy
from spectral.io.aviris import read_aviris_bands
from spectral.tests import testdir
from spectral.tests.spytest import SpyTest

ECOSTRESS_DATA_DIR = os.path.join(os.path.split(__file__)[0],
                                  'data/ecostress')
ECOSTRESS_DB = os.path.join(testdir, 'ecostress.db')
RELAB_DATA_DIR = os.path.join(os.path.split(__file__)[0],
                                  'data/relab/data/')
RELAB_DB = os.path.join(testdir, 'relab.db')
USGS_DATA_DIR = os.path.join(os.path.split(__file__)[0],
                             'data/usgs/ASCIIdata')
USGS_DB = os.path.join(testdir, 'usgs.db')
AVIRIS_BAND_FILE = os.path.join(os.path.split(__file__)[0], 'data/92AV3C.spc')


class ECOSTRESSDatabaseCreationTest(SpyTest):
    '''Tests ECOSTRESS database creation from text files.'''

    def __init__(self):
        pass

    def setup(self):
        if not os.path.isdir(testdir):
            os.makedirs(testdir)
        if os.path.exists(ECOSTRESS_DB):
            os.remove(ECOSTRESS_DB)

    def test_create_database(self):
        '''Test creating new database from ECOSTRESS data files.'''
        db = spy.EcostressDatabase.create(ECOSTRESS_DB,
                                          ECOSTRESS_DATA_DIR)
        assert (list(db.query('SELECT COUNT() FROM Spectra'))[0][0] == 3)


class ECOSTRESSDatabaseTest(SpyTest):
    '''Tests that ECOSTRESS database works properly'''

    def __init__(self):
        pass

    def setup(self):
        self.db = spy.EcostressDatabase(ECOSTRESS_DB)

    def test_read_signatures(self):
        '''Can get spectra from the opened database.'''
        assert (list(self.db.query('SELECT COUNT() FROM Spectra'))[0][0] == 3)

    def test_create_envi_lib(self):
        '''Can resample spectra and create an ENVI spectral library.'''
        bands = read_aviris_bands(AVIRIS_BAND_FILE)
        cursor = self.db.query('SELECT SpectrumID FROM Spectra')
        ids = [r[0] for r in cursor]
        bands.centers = [x / 1000. for x in bands.centers]
        bands.bandwidths = [x / 1000. for x in bands.bandwidths]
        slib = self.db.create_envi_spectral_library(ids, bands)
        assert (slib.spectra.shape == (3, 220))

class RELABDatabaseCreationTest(SpyTest):
    '''Tests RELAB database creation from text files.'''

    def __init__(self):
        pass

    def setup(self):
        if not os.path.isdir(testdir):
            os.makedirs(testdir)
        if os.path.exists(RELAB_DB):
            os.remove(RELAB_DB)

    def test_create_database(self):
        '''Test creating new database from RELAB data files.'''
        db = spy.RelabDatabase.create(RELAB_DB, RELAB_DATA_DIR)
        assert(list(db.query('SELECT COUNT() FROM Spectra'))[0][0] == 1)

class RELABDatabaseTest(SpyTest):
    '''Tests that RELAB database works properly'''

    def __init__(self):
        pass

    def setup(self):
        self.db = spy.RelabDatabase(RELAB_DB)

    def test_read_signatures(self):
        '''Can get spectra from the opened database.'''
        assert(list(self.db.query('SELECT COUNT() FROM Spectra'))[0][0] == 1)

class USGSDatabaseCreationTest(SpyTest):
    '''Tests USGS database creation from text files.'''

    def __init__(self):
        pass

    def setup(self):
        if not os.path.isdir(testdir):
            os.makedirs(testdir)
        if os.path.exists(USGS_DB):
            os.remove(USGS_DB)

    def test_create_database(self):
        '''Test creating new database from USGS data files.'''
        db = spy.USGSDatabase.create(USGS_DB, USGS_DATA_DIR)
        assert (list(db.query('SELECT COUNT() FROM Samples'))[0][0] == 8)
        assert (list(db.query('SELECT COUNT() FROM SpectrometerData'))
               [0][0] == 13)


class USGSDatabaseTest(SpyTest):
    '''Tests that USGS database works properly'''

    def __init__(self):
        pass

    def setup(self):
        self.db = spy.USGSDatabase(USGS_DB)

    def test_read_signatures(self):
        '''Can get spectra from the opened database.'''
        assert (list(self.db.query('SELECT COUNT() FROM Samples'))[0][0] == 8)
        assert (list(self.db.query('SELECT COUNT() FROM SpectrometerData'))
               [0][0] == 13)

        some_sample = list(self.db.query('''SELECT Chapter, FileName,
                    AssumedWLSpmeterDataID,
                    NumValues, MinValue, MaxValue
                    FROM Samples
                    WHERE LibName='liba' AND Record=1 AND
                    Description='Material a b0 0 ASDFRa AREF' AND
                    Spectrometer='ASDFR' AND Purity='a' AND MeasurementType='AREF'
                    '''))[0]
        assert (some_sample[0] == 'ChapterB_b0')
        assert (some_sample[1] == 'liba_Material_a_b0_0_ASDFRa_AREF.txt')
        assert (some_sample[3] == 24)
        assert_almost_equal(some_sample[4], 0.33387077)
        assert_almost_equal(some_sample[5], 0.51682192)

        some_spectrometer_data = list(self.db.query('''SELECT LibName, Record, MeasurementType, Unit,
                Name, Description, FileName, NumValues, MinValue, MaxValue
                FROM SpectrometerData
                WHERE SpectrometerDataID=?
                ''', (some_sample[2],)))[0]
        assert (some_spectrometer_data[0] == 'liba')
        assert (some_spectrometer_data[1] == 13)
        assert (some_spectrometer_data[2] == 'Wavelengths')
        assert (some_spectrometer_data[3] == 'micrometer')
        assert (some_spectrometer_data[4] == 'ASD')
        assert (some_spectrometer_data[5] == 'Wavelengths ASD 0.35-2.5 um')
        assert (some_spectrometer_data[6] ==
               'liba_Wavelengths_ASD_0.35-2.5_um.txt')
        assert (some_spectrometer_data[7] == 24)
        assert_almost_equal(some_spectrometer_data[8], 0.35)
        assert_almost_equal(some_spectrometer_data[9], 2.5)

    def test_get_spectrum(self):
        some_sample_id = list(self.db.query('''SELECT SampleID
            FROM Samples
            WHERE LibName='libc' AND Description='Material D 2 AVIRISb RTGC'
            '''))[0][0]
        (x, y) = self.db.get_spectrum(some_sample_id)
        assert (len(x) == len(y))
        assert (len(y) == 7)
        assert_almost_equal(y[0], 0.010381651)
        assert_almost_equal(x[-1], 2.2020326)

    def test_create_envi_lib(self):
        '''Can resample spectra and create an ENVI spectral library.'''
        bands = read_aviris_bands(AVIRIS_BAND_FILE)
        cursor = self.db.query('SELECT SampleID FROM Samples')
        ids = [r[0] for r in cursor]
        bands.centers = [x / 1000. for x in bands.centers]
        bands.bandwidths = [x / 1000. for x in bands.bandwidths]
        slib = self.db.create_envi_spectral_library(ids, bands)
        assert (slib.spectra.shape == (8, 220))


def run():
    print('\n' + '-' * 72)
    print('Running database tests.')
    print('-' * 72)
    for T in [ECOSTRESSDatabaseCreationTest, ECOSTRESSDatabaseTest, \
            RELABDatabaseCreationTest, RELABDatabaseTest, \
            USGSDatabaseCreationTest, USGSDatabaseTest]:
        T().run()


if __name__ == '__main__':
    from spectral.tests.run import parse_args, reset_stats, print_summary
    import logging
    logging.getLogger('spectral').setLevel(logging.ERROR)
    parse_args()
    reset_stats()
    run()
    print_summary()
