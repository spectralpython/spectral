'''
Tests of functions associated with spectral databases.

Note that the ASTER, ECOSTRESS, RELAB, and USGS databases are built from
bundled sample data under `spectral/tests/data/`, so these tests do not
depend on the external SPECTRAL_DATA sample set and always run.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_database.py
'''

import os

from numpy.testing import assert_almost_equal
import pytest

import spectral as spy
from spectral.io.aviris import read_aviris_bands

ASTER_DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data/aster')
ECOSTRESS_DATA_DIR = os.path.join(os.path.split(__file__)[0],
                                  'data/ecostress')
RELAB_DATA_DIR = os.path.join(os.path.split(__file__)[0],
                                  'data/relab/data/')
USGS_DATA_DIR = os.path.join(os.path.split(__file__)[0],
                             'data/usgs/ASCIIdata')
AVIRIS_BAND_FILE = os.path.join(os.path.split(__file__)[0], 'data/92AV3C.spc')


class TestAsterDatabase:
    '''Tests ASTER database creation and querying.

    The two bundled sample files were crafted to exercise both branches of
    `_import_files`'s solid/liquid `phase` determination and its "y units"/
    "measurement" text-normalization logic (one file reflectance/solid, the
    other transmittance/liquid), in addition to the basic create/query
    round trip.
    '''

    @pytest.fixture(scope='class')
    def db(self, testdir):
        db_file = os.path.join(testdir, 'aster.db')
        spy.AsterDatabase.create(db_file, ASTER_DATA_DIR)
        return spy.AsterDatabase(db_file)

    def test_create_database(self, db):
        '''Test creating new database from ASTER data files.'''
        assert (list(db.query('SELECT COUNT() FROM Samples'))[0][0] == 2)
        assert (list(db.query('SELECT COUNT() FROM Spectra'))[0][0] == 2)

    def test_solid_sample_phase_and_units(self, db):
        sample = list(db.query('''SELECT Phase, Name, Type, Class, SubClass,
            ParticleSize, SampleNum, Owner, Origin, Description
            FROM Samples WHERE Name='test mineral one\''''))[0]
        assert sample[0] == 'solid'
        assert sample[2] == 'mineral'
        assert sample[3] == 'silicate'
        assert sample[4] == 'tectosilicate'
        assert sample[5] == 'solid'
        assert sample[6] == 'tst001'
        assert sample[7] == 'test owner'
        assert sample[8] == 'test origin facility continued origin text'
        assert 'line six' in sample[9]

        spectrum = list(db.query('''SELECT Measurement, XUnit, YUnit,
            MinWavelength, MaxWavelength, NumValues
            FROM Spectra, Samples
            WHERE Spectra.SampleID = Samples.SampleID
            AND Samples.Name='test mineral one\''''))[0]
        assert spectrum[0] == 'reflectance'
        assert spectrum[1] == 'wavelength (micrometers)'
        assert spectrum[2] == 'reflectance (percent)'
        assert_almost_equal(spectrum[3], 0.4)
        assert_almost_equal(spectrum[4], 2.0)
        assert spectrum[5] == 5

    def test_liquid_sample_phase_and_units(self, db):
        '''The bundled "liquid" sample verifies the fixed `phase`
        determination (previously always resulted in "solid" due to a
        missing `()` on `.lower`) and the "measurement"/"y units"
        transmittance-normalization branches.'''
        sample = list(db.query('''SELECT Phase FROM Samples
            WHERE Name='test liquid sample\''''))[0]
        assert sample[0] == 'liquid'

        spectrum = list(db.query('''SELECT Measurement, YUnit
            FROM Spectra, Samples
            WHERE Spectra.SampleID = Samples.SampleID
            AND Samples.Name='test liquid sample\''''))[0]
        assert spectrum[0] == 'transmittance'
        assert spectrum[1] == 'transmittance (percent)'

    def test_get_spectrum(self, db):
        some_sample_id = list(db.query('''SELECT SampleID FROM Samples
            WHERE Name='test mineral one\''''))[0][0]
        spectrum_id = list(db.query(
            'SELECT SpectrumID FROM Spectra WHERE SampleID=?',
            (some_sample_id,)))[0][0]
        (x, y) = db.get_spectrum(spectrum_id)
        assert (len(x) == len(y) == 5)
        assert_almost_equal(x[0], 0.4)
        assert_almost_equal(y[-1], 50.0)

    def test_get_spectrum_not_found_raises(self, db):
        with pytest.raises(Exception):
            db.get_spectrum(-1)

    def test_get_signature(self, db):
        some_sample_id = list(db.query('''SELECT SampleID FROM Samples
            WHERE Name='test mineral one\''''))[0][0]
        spectrum_id = list(db.query(
            'SELECT SpectrumID FROM Spectra WHERE SampleID=?',
            (some_sample_id,)))[0][0]
        sig = db.get_signature(spectrum_id)
        assert sig.measurement_id == spectrum_id
        assert sig.sample_name == 'test mineral one'
        assert sig.sample_id == some_sample_id
        assert_almost_equal(sig.x[0], 0.4)
        assert_almost_equal(sig.y[-1], 50.0)

    def test_get_signature_not_found_raises(self, db):
        with pytest.raises(Exception):
            db.get_signature(-1)

    def test_create_envi_lib(self, db):
        '''Can resample spectra and create an ENVI spectral library.'''
        bands = read_aviris_bands(AVIRIS_BAND_FILE)
        cursor = db.query('SELECT SpectrumID FROM Spectra')
        ids = [r[0] for r in cursor]
        bands.centers = [x / 1000. for x in bands.centers]
        bands.bandwidths = [x / 1000. for x in bands.bandwidths]
        slib = db.create_envi_spectral_library(ids, bands)
        assert (slib.spectra.shape == (2, 220))

    def test_create_raises_if_file_exists(self, db, testdir):
        with pytest.raises(Exception):
            spy.AsterDatabase.create(os.path.join(testdir, 'aster.db'),
                                     ASTER_DATA_DIR)

    def test_create_raises_for_invalid_data_dir(self, testdir):
        db_file = os.path.join(testdir, 'aster_bad_dir.db')
        with pytest.raises(Exception):
            spy.AsterDatabase.create(db_file, '/not/a/real/directory')


class TestECOSTRESSDatabase:
    '''Tests ECOSTRESS database creation and querying.'''

    @pytest.fixture(scope='class')
    def db(self, testdir):
        db_file = os.path.join(testdir, 'ecostress.db')
        spy.EcostressDatabase.create(db_file, ECOSTRESS_DATA_DIR)
        return spy.EcostressDatabase(db_file)

    def test_create_database(self, db):
        '''Test creating new database from ECOSTRESS data files.'''
        assert (list(db.query('SELECT COUNT() FROM Spectra'))[0][0] == 3)

    def test_read_signatures(self, db):
        '''Can get spectra from the opened database.'''
        assert (list(db.query('SELECT COUNT() FROM Spectra'))[0][0] == 3)

    def test_create_envi_lib(self, db):
        '''Can resample spectra and create an ENVI spectral library.'''
        bands = read_aviris_bands(AVIRIS_BAND_FILE)
        cursor = db.query('SELECT SpectrumID FROM Spectra')
        ids = [r[0] for r in cursor]
        bands.centers = [x / 1000. for x in bands.centers]
        bands.bandwidths = [x / 1000. for x in bands.bandwidths]
        slib = db.create_envi_spectral_library(ids, bands)
        assert (slib.spectra.shape == (3, 220))


class TestRELABDatabase:
    '''Tests RELAB database creation and querying.'''

    @pytest.fixture(scope='class')
    def db(self, testdir):
        db_file = os.path.join(testdir, 'relab.db')
        spy.RelabDatabase.create(db_file, RELAB_DATA_DIR)
        return spy.RelabDatabase(db_file)

    def test_create_database(self, db):
        '''Test creating new database from RELAB data files.'''
        assert (list(db.query('SELECT COUNT() FROM Spectra'))[0][0] == 8)

    def test_read_signatures(self, db):
        '''Can get spectra from the opened database.'''
        assert (list(db.query('SELECT COUNT() FROM Spectra'))[0][0] == 8)


class TestUSGSDatabase:
    '''Tests USGS database creation and querying.'''

    @pytest.fixture(scope='class')
    def db(self, testdir):
        db_file = os.path.join(testdir, 'usgs.db')
        spy.USGSDatabase.create(db_file, USGS_DATA_DIR)
        return spy.USGSDatabase(db_file)

    def test_create_database(self, db):
        '''Test creating new database from USGS data files.'''
        assert (list(db.query('SELECT COUNT() FROM Samples'))[0][0] == 8)
        assert (list(db.query('SELECT COUNT() FROM SpectrometerData'))
               [0][0] == 13)

    def test_read_signatures(self, db):
        '''Can get spectra from the opened database.'''
        assert (list(db.query('SELECT COUNT() FROM Samples'))[0][0] == 8)
        assert (list(db.query('SELECT COUNT() FROM SpectrometerData'))
               [0][0] == 13)

        some_sample = list(db.query('''SELECT Chapter, FileName,
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

        some_spectrometer_data = list(db.query('''SELECT LibName, Record, MeasurementType, Unit,
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

    def test_get_spectrum(self, db):
        some_sample_id = list(db.query('''SELECT SampleID
            FROM Samples
            WHERE LibName='libc' AND Description='Material D 2 AVIRISb RTGC'
            '''))[0][0]
        (x, y) = db.get_spectrum(some_sample_id)
        assert (len(x) == len(y))
        assert (len(y) == 7)
        assert_almost_equal(y[0], 0.010381651)
        assert_almost_equal(x[-1], 2.2020326)

    def test_create_envi_lib(self, db):
        '''Can resample spectra and create an ENVI spectral library.'''
        bands = read_aviris_bands(AVIRIS_BAND_FILE)
        cursor = db.query('SELECT SampleID FROM Samples')
        ids = [r[0] for r in cursor]
        bands.centers = [x / 1000. for x in bands.centers]
        bands.bandwidths = [x / 1000. for x in bands.bandwidths]
        slib = db.create_envi_spectral_library(ids, bands)
        assert (slib.spectra.shape == (8, 220))
