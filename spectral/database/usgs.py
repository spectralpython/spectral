'''
Code for reading and managing USGS spectral library data.

References:
    Kokaly, R.F., Clark, R.N., Swayze, G.A., Livo, K.E., Hoefen, T.M., Pearson,
    N.C., Wise, R.A., Benzel, W.M., Lowers, H.A., Driscoll, R.L., and Klein, A.J.,
    2017, USGS Spectral Library Version 7: U.S. Geological Survey Data Series 1035,
    61 p., https://doi.org/10.3133/ds1035.
'''

from __future__ import absolute_import, division, print_function, unicode_literals
from spectral.utilities.python23 import IS_PYTHON3, tobytes, frombytes
from .spectral_database import SpectralDatabase

import re
import logging

if IS_PYTHON3:
    def readline(fin): return fin.readline()
    def open_file(filename): return open(filename, encoding='iso-8859-1')
else:
    def readline(fin): return fin.readline().decode('iso-8859-1')
    def open_file(filename): return open(filename)

table_schemas = [
    'CREATE TABLE Samples (SampleID INTEGER PRIMARY KEY, LibName TEXT, Record INTEGER, '
    'Description TEXT, Spectrometer TEXT, Purity TEXT, MeasurementType TEXT, Chapter TEXT, FileName TEXT, '
    'AssumedWLSpmeterDataID INTEGER, '
    'NumValues INTEGER, MinValue FLOAT, MaxValue FLOAT, ValuesArray BLOB)',
    'CREATE TABLE SpectrometerData (SpectrometerDataID INTEGER PRIMARY KEY, LibName TEXT, '
    'Record INTEGER, MeasurementType TEXT, Unit TEXT, Name TEXT, FullName TEXT, Description TEXT, FileName TEXT, '
    'NumValues INTEGER, MinValue FLOAT, MaxValue FLOAT, ValuesArray BLOB)'
]

arraytypecode = chr(ord('f'))


def _get_pure_spectrometer_name(name):
    # Split 'AVIRIS13' into ['', 'AVIRIS', '13', ''].
    # Split 'BECK' into ['BECK'], 'NIC4' into ['NIC4'].
    splitted = re.split('([A-Z]+)([0-9][0-9]+)', name)
    return splitted[1] if len(splitted) > 1 else splitted[0]


class SpectrometerData:
    '''
        Holds data for spectrometer, from USGS spectral library.
    '''

    def __init__(self, libname, record, measurement_type, unit, spectrometer_name, full_name,
                 description, file_name, values):
        self.libname = libname
        self.record = record
        self.measurement_type = measurement_type
        self.unit = unit
        self.spectrometer_name = spectrometer_name
        self.full_name = full_name
        self.description = description
        self.file_name = file_name
        self.values = values

    def header(self):
        '''
            Returns:
                String representation of basic meta data.
        '''
        return '{} Record={}: {} {} {}'.format(self.libname, self.record,  self.measurement,
                                               self.full_name, self.description)

    @ classmethod
    def read_from_file(cls, filename):
        '''
            Constructs SpectrometerData from file.

            Arguments:

                `filename` (str):

                    Path to file containing data.

            Returns:
                A `SpectrometerData` constructed from data parsed from file.
        '''
        import os
        logger = logging.getLogger('spectral')
        with open_file(filename) as f:
            header_line = readline(f)
            libname, record, measurement_type, unit, spectrometer_name, full_name, description = \
                SpectrometerData._parse_header(header_line.strip())

            values = []
            for line in f:
                if len(line) == 0:
                    break
                try:
                    values.append(float(line.strip()))
                except:
                    logger.error('In file %s found unparsable line.', filename)

            file_name = os.path.basename(filename)
            return cls(libname, record, measurement_type, unit, spectrometer_name, full_name, description, file_name, values)

    @ staticmethod
    def _do_replacements(header_line):
        header_line = re.sub(r'SRF Band ([0-9]+)', r'SRF_Band_\1', header_line)
        header_line = re.sub(r'AVIRIS ([0-9]+)', r'AVIRIS\1', header_line)
        header_line = header_line.replace('Bandpass (FWHM)', 'Bandpass_FWHM')
        return header_line

    @ staticmethod
    def _assume_unit(header_line):
        if re.search(r'\bnm\b', header_line) is not None:
            return 'nanometer'
        if 'nanometer' in header_line:
            return 'nanometer'
        # 'um', 'microns' are usually found in these files, but this is default
        # anyway.
        return 'micrometer'

    @ staticmethod
    def _parse_header(header_line):
        # It is difficult to parse this data,
        # things are separated by spaces, but inside of what should be single datum,
        # there are spaces, so only human can get it right.
        header_line = SpectrometerData._do_replacements(header_line)
        elements = header_line.split()

        libname = elements[0]

        # From 'Record=1234:' extract 1234.
        record = int(elements[1].split('=')[1][:-1])

        measurement_type = elements[2]
        full_name = elements[3]
        spectrometer_name = _get_pure_spectrometer_name(full_name)
        description = ' '.join(elements[4:])

        unit = SpectrometerData._assume_unit(header_line)

        return libname, record, measurement_type, unit, spectrometer_name, full_name, description


class SampleData:
    '''
        Holds parsed data for single sample from USGS spectral library.
    '''

    def __init__(self, libname=None, record=None, description=None, spectrometer=None,
                 purity=None, measurement_type=None, chapter=None, file_name=None, values=None):
        self.libname = libname
        self.record = record
        self.description = description
        self.spectrometer = spectrometer
        self.purity = purity
        self.measurement_type = measurement_type
        self.chapter = chapter
        self.file_name = file_name
        self.values = values

    def header(self):
        '''
            Returns:
                String representation of basic meta data.
        '''
        return '{} Record={}: {} {}{} {}'.format(self.libname, self.record,
                                                 self.description, self.spectrometer,
                                                 self.purity, self.measurement_type)

    @staticmethod
    def _parse_header(header_line):
        elements = header_line.split()

        libname = elements[0]

        # From 'Record=1234:' extract 1234.
        record = int(elements[1].split('=')[1][:-1])

        # Join everything between record and spectrometer into description.
        description = ' '.join(elements[2:-2])

        # Split 'AVIRIS13aa' into ['', 'AVIRIS13', 'aa', ''].
        smpurity = re.split('([A-Z0-9]+)([a-z]+)', elements[-2])
        spectrometer = smpurity[1]
        purity = smpurity[2]

        measurement_type = elements[-1]

        return libname, record, description, spectrometer, purity, measurement_type

    @classmethod
    def read_from_file(cls, filename, chapter=None):
        '''
            Constructs SampleData from file.

            Arguments:

                `filename` (str):

                    Path to file containing data.

            Returns:
                A `SampleData` constructed from data parsed from file.
        '''
        import os
        logger = logging.getLogger('spectral')
        with open(filename) as f:
            header_line = f.readline()
            libname, record, description, spectrometer, purity, measurement_type = \
                SampleData._parse_header(header_line.strip())

            values = []
            for line in f:
                if len(line) == 0:
                    break
                try:
                    values.append(float(line.strip()))
                except:
                    logger.error('In file %s found unparsable line.', filename)

            file_name = os.path.basename(filename)
            return cls(libname, record, description, spectrometer, purity,
                       measurement_type, chapter, file_name, values)


class USGSDatabase(SpectralDatabase):
    '''A relational database to manage USGS spectral library data.'''
    schemas = table_schemas

    def _assume_wavelength_spectrometer_data_id(self, sampleData):
        # We can't know this for sure, but these heuristics haven't failed so far.

        # Prepare paramters.
        # These parameters are mandatory to match.
        libname = sampleData.libname
        num_values = len(sampleData.values)
        # Spectrometer might not match in subdirectories where data is convolved
        # or resampled. In other directories, without spectrometer there is
        # few possible choices, so spectrometer isolates the one we need.
        spectrometer = sampleData.spectrometer

        logger = logging.getLogger('spectral')

        # Start with the most specific.
        query = '''SELECT SpectrometerDataID FROM SpectrometerData WHERE
                    MeasurementType = 'Wavelengths' AND LibName = ? AND NumValues = ?
                    AND Name = ?'''
        result = self.cursor.execute(
            query, (libname, num_values, spectrometer))
        rows = result.fetchall()
        if len(rows) == 0:
            query = '''SELECT SpectrometerDataID FROM SpectrometerData WHERE
            MeasurementType = 'Wavelengths' AND LibName = ? AND NumValues = ?
            AND Name LIKE ?'''
            result = self.cursor.execute(
                # ASDFR -> ASD, and '%' just to be sure.
                query, (libname, num_values, spectrometer[:3] + '%'))
            rows = result.fetchall()
        if len(rows) >= 1:
            if len(rows) > 1:
                logger.warning('Found multiple spectrometers with measurement_type Wavelengths, '
                               ' LibName %s, NumValues %d and Name %s', libname, num_values, spectrometer)
            return rows[0][0]

        # Try to be less specific without spectrometer name.
        query = '''SELECT SpectrometerDataID FROM SpectrometerData WHERE
            MeasurementType = 'Wavelengths' AND LibName = ? AND NumValues = ?'''
        result = self.cursor.execute(query, (libname, num_values))
        rows = result.fetchall()
        if len(rows) < 1:
            raise Exception('Wavelengths for spectrometer not found, for LibName = {} and NumValues = {}, from file {}'.format(
                libname, num_values, sampleData.file_name))
        if len(rows) > 1:
            logger.warning('Found multiple spectrometers with measurement_type Wavelengths, '
                           ' LibName %s and NumValues %d, from file %s', libname, num_values, sampleData.file_name)
        return rows[0][0]

    def _add_sample_data(self, spdata):
        import sqlite3
        import array
        sql = '''INSERT INTO Samples (LibName, Record,
                    Description, Spectrometer, Purity, MeasurementType, Chapter, FileName,
                    AssumedWLSpmeterDataID,
                    NumValues, MinValue, MaxValue, ValuesArray)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        values = sqlite3.Binary(
            tobytes(array.array(arraytypecode, spdata.values)))
        num_values = len(spdata.values)
        min_value = min(spdata.values)
        max_value = max(spdata.values)
        assumedWLSpmeterDataID = self._assume_wavelength_spectrometer_data_id(
            spdata)
        self.cursor.execute(sql, (spdata.libname, spdata.record, spdata.description,
                                  spdata.spectrometer, spdata.purity, spdata.measurement_type,
                                  spdata.chapter, spdata.file_name, assumedWLSpmeterDataID,
                                  num_values, min_value, max_value, values))
        rowId = self.cursor.lastrowid
        self.db.commit()
        return rowId

    def _add_spectrometer_data(self, spdata):
        import sqlite3
        import array
        sql = '''INSERT INTO SpectrometerData (LibName, Record, MeasurementType, Unit,
                Name, FullName, Description, FileName, NumValues, MinValue, MaxValue, ValuesArray)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        values = sqlite3.Binary(
            tobytes(array.array(arraytypecode, spdata.values)))
        num_values = len(spdata.values)
        min_value = min(spdata.values)
        max_value = max(spdata.values)
        self.cursor.execute(
            sql, (spdata.libname, spdata.record, spdata.measurement_type, spdata.unit,
                  spdata.spectrometer_name, spdata.full_name, spdata.description,
                  spdata.file_name, num_values, min_value, max_value, values))
        rowId = self.cursor.lastrowid
        self.db.commit()
        return rowId

    @classmethod
    def create(cls, filename, usgs_data_dir=None):
        '''Creates an USGS relational database by parsing USGS data files.

        Arguments:

            `filename` (str):

                Name of the new sqlite database file to create.

            `usgs_data_dir` (str):

                Path to the USGS ASCII data directory. This directory should
                contain subdirectories, which containes chapter directories.
                E.g. if provided `usgs_data_dir` is '/home/user/usgs/ASCIIdata',
                then relative path to single sample could be
                'ASCIIdata_splib07b/ChapterL_Liquids/splib07b_H2O-Ice_GDS136_77K_BECKa_AREF.txt'
                If this argument is not provided, no data will be imported.

        Returns:

            An :class:`~spectral.database.USGSDatabase` object.

        Example::

            >>> USGSDatabase.create("usgs_lib.db", "/home/user/usgs/ASCIIdata")

        This is a class method (it does not require instantiating an
        USGSDatabase object) that creates a new database by parsing files in the
        USGS library ASCIIdata directory.  Normally, this should only
        need to be called once. Subsequently, a corresponding database object
        can be created by instantiating a new USGSDatabase object with the
        path the database file as its argument.  For example::

            >>> from spectral.database.usgs import USGSDatabase
            >>> db = USGSDatabase("usgs_lib.db")
        '''
        import os
        if os.path.isfile(filename):
            raise Exception('Error: Specified file already exists.')
        db = cls()
        db._connect(filename)
        for schema in cls.schemas:
            db.cursor.execute(schema)
        if usgs_data_dir:
            db._import_files(usgs_data_dir)
        return db

    def __init__(self, sqlite_filename=None):
        '''Creates a database object to interface an existing database.

        Arguments:

            `sqlite_filename` (str):

                Name of the database file.  If this argument is not provided,
                an interface to a database file will not be established.

        Returns:

            An :class:`~spectral.USGSDatabase` connected to the database.
        '''
        from spectral.io.spyfile import find_file_path
        if sqlite_filename:
            self._connect(find_file_path(sqlite_filename))
        else:
            self.db = None
            self.cursor = None

    def _import_files(self, data_dir):
        from glob import glob
        import numpy
        import os
        logger = logging.getLogger('spectral')

        if not os.path.isdir(data_dir):
            raise Exception('Error: Invalid directory name specified.')

        num_sample_files = 0
        num_spectrometer_files = 0

        for sublib in os.listdir(data_dir):
            sublib_dir = os.path.join(data_dir, sublib)
            if not os.path.isdir(sublib_dir):
                continue

            # Process instrument data one by one.
            for f in glob(sublib_dir + '/*.txt'):
                logger.info('Importing spectrometer file %s', f)
                spdata = SpectrometerData.read_from_file(f)
                self._add_spectrometer_data(spdata)
                num_spectrometer_files += 1

            # Go into each chapter directory and process individual samples.
            for chapter in os.listdir(sublib_dir):
                chapter_dir = os.path.join(sublib_dir, chapter)
                if not os.path.isdir(chapter_dir):
                    continue

                for f in glob(chapter_dir + '/*.txt'):
                    logger.info('Importing sample file %s', f)
                    spdata = SampleData.read_from_file(f, chapter)
                    self._add_sample_data(spdata)
                    num_sample_files += 1

        logger.info('Imported %d sample files and %d spectrometer files.',
                    num_sample_files, num_spectrometer_files)

    def get_spectrum(self, sampleID):
        '''Returns a spectrum from the database.

        Usage:

            (x, y) = usgs.get_spectrum(sampleID)

        Arguments:

            `sampleID` (int):

                The **SampleID** value for the desired spectrum from the
                **Samples** table in the database.

        Returns:

            `x` (list):

                Band centers for the spectrum.
                This is extraced from assumed spectrometer for given sample.

            `y` (list):

                Spectrum data values for each band.

        Returns a pair of vectors containing the wavelengths and measured
        values values of a measurment.
        '''
        import array
        query = '''SELECT ValuesArray, AssumedWLSpmeterDataID FROM Samples WHERE SampleID = ?'''
        result = self.cursor.execute(query, (sampleID,))
        rows = result.fetchall()
        if len(rows) < 1:
            raise Exception('Measurement record not found.')
        y = array.array(arraytypecode)
        frombytes(y, rows[0][0])
        assumedWLSpmeterDataID = rows[0][1]

        query = '''SELECT ValuesArray FROM SpectrometerData WHERE SpectrometerDataID = ?'''
        result = self.cursor.execute(
            query, (assumedWLSpmeterDataID,))
        rows = result.fetchall()
        if len(rows) < 1:
            raise Exception('Measurement (wavelengths) record not found.')
        x = array.array(arraytypecode)
        frombytes(x, rows[0][0])

        return (list(x), list(y))
