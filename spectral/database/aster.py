#########################################################################
#
#   aster.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2010 Thomas Boggs
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
'''Code for reading and managing ASTER spectral library data.'''

from __future__ import division, print_function, unicode_literals

from spectral.utilities.python23 import IS_PYTHON3, tobytes, frombytes

if IS_PYTHON3:
    readline = lambda fin: fin.readline()
    open_file = lambda filename: open(filename, encoding='iso-8859-1')
else:
    readline = lambda fin: fin.readline().decode('iso-8859-1')
    open_file = lambda filename: open(filename)

table_schemas = [
    'CREATE TABLE Samples (SampleID INTEGER PRIMARY KEY, Name TEXT, Type TEXT, Class TEXT, SubClass TEXT, '
    'ParticleSize TEXT, SampleNum TEXT, Owner TEXT, Origin TEXT, Phase TEXT, Description TEXT)',
    'CREATE TABLE Spectra (SpectrumID INTEGER PRIMARY KEY, SampleID INTEGER, SensorCalibrationID INTEGER, '
    'Instrument TEXT, Environment TEXT, Measurement TEXT, '
    'XUnit TEXT, YUnit TEXT, MinWavelength FLOAT, MaxWavelength FLOAT, '
    'NumValues INTEGER, XData BLOB, YData BLOB)',
]

arraytypecode = chr(ord('f'))

# These files contained malformed signature data and will be ignored.
bad_files = [
    'jhu.nicolet.mineral.silicate.tectosilicate.fine.albite1.spectrum.txt',
    'usgs.perknic.rock.igneous.mafic.colid.me3.spectrum.txt'
]


def read_pair(fin, num_lines=1):
    '''Reads a colon-delimited attribute-value pair from the file stream.'''
    s = ''
    for i in range(num_lines):
        s += " " + readline(fin).strip()
    return [x.strip().lower() for x in s.split(':')]


class Signature:
    '''Object to store sample/measurement metadata, as well as wavelength-signatrure vectors.'''
    def __init__(self):
        self.sample = {}
        self.measurement = {}


def read_aster_file(filename):
    '''Reads an ASTER 2.x spectrum file.'''
    fin = open_file(filename)

    s = Signature()

    # Number of lines per metadata attribute value
    lpv = [1] * 8 + [2] + [6]

    # A few files have an additional "Colleted by" sample metadata field, which
    # sometimes affects the number of header lines

    haveCollectedBy = False
    for i in range(30):
        line = readline(fin).strip()
        if line.find('Collected by:') >= 0:
            haveCollectedBy = True
            collectedByLineNum = i
        if line.startswith('Description:'):
            descriptionLineNum = i
        if line.startswith('Measurement:'):
            measurementLineNum = i

    if haveCollectedBy:
        lpv = [1] * 10 + [measurementLineNum - descriptionLineNum]

    # Read sample metadata
    fin.seek(0)
    for i in range(len(lpv)):
        pair = read_pair(fin, lpv[i])
        s.sample[pair[0].lower()] = pair[1]

    # Read measurement metadata
    lpv = [1] * 8 + [2]
    for i in range(len(lpv)):
        pair = read_pair(fin, lpv[i])
        if len(pair) < 2:
            print(pair)
        s.measurement[pair[0].lower()] = pair[1]

    # Read signature spectrum
    pairs = []
    for line in fin.readlines():
        line = line.strip()
        if len(line) == 0:
            continue
        pair = line.split()
        nItems = len(pair)

        # Try to handle invalid values on signature lines
        if nItems == 1:
#           print 'single item (%s) on signature line, %s' \
#                 %  (pair[0], filename)
            continue
        elif nItems > 2:
            print('more than 2 values on signature line,', filename)
            continue
        try:
            x = float(pair[0])
        except:
            print('corrupt signature line,', filename)
        if x == 0:
#           print 'Zero wavelength value', filename
            continue
        elif x < 0:
            print('Negative wavelength value,', filename)
            continue

        pairs.append(pair)

    [x, y] = [list(v) for v in zip(*pairs)]

    # Make sure wavelengths are ascending
    if float(x[0]) > float(x[-1]):
        x.reverse()
        y.reverse()
    s.x = [float(val) for val in x]
    s.y = [float(val) for val in y]
    s.measurement['first x value'] = x[0]
    s.measurement['last x value'] = x[-1]
    s.measurement['number of x values'] = len(x)

    fin.close()
    return s


class AsterDatabase:
    '''A relational database to manage ASTER spectral library data.'''
    schemas = table_schemas

    def _add_sample(self, name, sampleType, sampleClass, subClass,
                    particleSize, sampleNumber, owner, origin, phase,
                    description):
        sql = '''INSERT INTO Samples (Name, Type, Class, SubClass, ParticleSize, SampleNum, Owner, Origin, Phase, Description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        self.cursor.execute(sql, (name, sampleType, sampleClass, subClass,
                                  particleSize, sampleNumber, owner, origin,
                                  phase, description))
        rowId = self.cursor.lastrowid
        self.db.commit()
        return rowId

    def _add_signature(
        self, sampleID, calibrationID, instrument, environment, measurement,
            xUnit, yUnit, minWavelength, maxWavelength, xData, yData):
        import sqlite3
        import array
        sql = '''INSERT INTO Spectra (SampleID, SensorCalibrationID, Instrument,
                 Environment, Measurement, XUnit, YUnit, MinWavelength, MaxWavelength,
                 NumValues, XData, YData) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        xBlob = sqlite3.Binary(tobytes(array.array(arraytypecode, xData)))
        yBlob = sqlite3.Binary(tobytes(array.array(arraytypecode, yData)))
        numValues = len(xData)
        self.cursor.execute(
            sql, (
                sampleID, calibrationID, instrument, environment, measurement,
                xUnit, yUnit, minWavelength, maxWavelength, numValues, xBlob,
                yBlob))
        rowId = self.cursor.lastrowid
        self.db.commit()
        return rowId

    @classmethod
    def create(cls, filename, aster_data_dir=None):
        '''Creates an ASTER relational database by parsing ASTER data files.

        Arguments:

            `filename` (str):

                Name of the new sqlite database file to create.

            `aster_data_dir` (str):

                Path to the directory containing ASTER library data files. If
                this argument is not provided, no data will be imported.

        Returns:

            An :class:`~spectral.database.AsterDatabase` object.

        Example::

            >>> AsterDatabase.create("aster_lib.db", "/CDROM/ASTER2.0/data")

        This is a class method (it does not require instantiating an
        AsterDatabase object) that creates a new database by parsing all of the
        files in the ASTER library data directory.  Normally, this should only
        need to be called once.  Subsequently, a corresponding database object
        can be created by instantiating a new AsterDatabase object with the
        path the database file as its argument.  For example::

            >>> from spectral.database.aster import AsterDatabase
            >>> db = AsterDatabase("aster_lib.db")
        '''
        import os
        if os.path.isfile(filename):
            raise Exception('Error: Specified file already exists.')
        db = cls()
        db._connect(filename)
        for schema in cls.schemas:
            db.cursor.execute(schema)
        if aster_data_dir:
            db._import_files(aster_data_dir)
        return db

    def __init__(self, sqlite_filename=None):
        '''Creates a database object to interface an existing database.

        Arguments:

            `sqlite_filename` (str):

                Name of the database file.  If this argument is not provided,
                an interface to a database file will not be established.

        Returns:

            An :class:`~spectral.AsterDatabase` connected to the database.
        '''
        from spectral.io.spyfile import find_file_path
        if sqlite_filename:
            self._connect(find_file_path(sqlite_filename))
        else:
            self.db = None
            self.cursor = None

    def read_file(self, filename):
        return read_aster_file(filename)

    def _import_files(self, data_dir, ignore=bad_files):
        '''Read each file in the ASTER library and convert to AVIRIS bands.'''
        from glob import glob
        import numpy
        import os

        if not os.path.isdir(data_dir):
            raise Exception('Error: Invalid directory name specified.')
        if ignore is not None:
            filesToIgnore = [data_dir + '/' + f for f in ignore]
        else:
            filesToIgnore = []

        numFiles = 0
        numIgnored = 0

        sigID = 1

        class Sig:
            pass
        sigs = []

        for f in glob(data_dir + '/*spectrum.txt'):
            if f in filesToIgnore:
                numIgnored += 1
                continue
            print('Importing %s.' % f)
            numFiles += 1
            sig = self.read_file(f)
            s = sig.sample
            if s['particle size'].lower == 'liquid':
                phase = 'liquid'
            else:
                phase = 'solid'
            if 'sample no.' in s:
                sampleNum = s['sample no.']
            else:
                sampleNum = ''
            id = self._add_sample(
                s['name'], s['type'], s['class'], s[
                    'subclass'], s['particle size'],
                sampleNum, s['owner'], s['origin'], phase, s['description'])

            instrument = os.path.basename(f).split('.')[1]
            environment = 'lab'
            m = sig.measurement

            # Correct numerous mispellings of "reflectance" and "transmittance"
            yUnit = m['y units']
            if yUnit.find('reflectence') > -1:
                yUnit = 'reflectance (percent)'
            elif yUnit.find('trans') == 0:
                yUnit = 'transmittance (percent)'
            measurement = m['measurement']
            if measurement[0] == 't':
                measurement = 'transmittance'
            self._add_signature(id, -1, instrument, environment, measurement,
                                m['x units'], yUnit, m['first x value'],
                                m['last x value'], sig.x, sig.y)
        if numFiles == 0:
            print('No data files were found in directory "%s".' \
                  % data_dir)
        else:
            print('Processed %d files.' % numFiles)
        if numIgnored > 0:
            print('Ignored the following %d bad files:' % (numIgnored))
            for f in filesToIgnore:
                print('\t' + f)

        return sigs

    def _connect(self, sqlite_filename):
        '''Establishes a connection to the Specbase sqlite database.'''
        import sqlite3
        self.db = sqlite3.connect(sqlite_filename)
        self.cursor = self.db.cursor()

    def get_spectrum(self, spectrumID):
        '''Returns a spectrum from the database.

        Usage:

            (x, y) = aster.get_spectrum(spectrumID)

        Arguments:

            `spectrumID` (int):

                The **SpectrumID** value for the desired spectrum from the
                **Spectra** table in the database.

        Returns:

            `x` (list):

                Band centers for the spectrum.

            `y` (list):

                Spectrum data values for each band.

        Returns a pair of vectors containing the wavelengths and measured
        values values of a measurment.  For additional metadata, call
        "get_signature" instead.
        '''
        import array
        query = '''SELECT XData, YData FROM Spectra WHERE SpectrumID = ?'''
        result = self.cursor.execute(query, (spectrumID,))
        rows = result.fetchall()
        if len(rows) < 1:
            raise 'Measurement record not found'
        x = array.array(arraytypecode)
        frombytes(x, rows[0][0])
        y = array.array(arraytypecode)
        frombytes(y, rows[0][1])
        return (list(x), list(y))

    def get_signature(self, spectrumID):
        '''Returns a spectrum with some additional metadata.

        Usage::

            sig = aster.get_signature(spectrumID)

        Arguments:

            `spectrumID` (int):

                The **SpectrumID** value for the desired spectrum from the
                **Spectra** table in the database.

        Returns:

            `sig` (:class:`~spectral.database.aster.Signature`):

                An object with the following attributes:

                ==============  =====   ========================================
                Attribute       Type            Description
                ==============  =====   ========================================
                measurement_id  int     SpectrumID value from Spectra table
                sample_name     str     **Sample** from the **Samples** table
                sample_id       int     **SampleID** from the **Samples** table
                x               list    list of band center wavelengths
                y               list    list of spectrum values for each band
                ==============  =====   ========================================
        '''
        import array

        # Retrieve spectrum from Spectra table
        query = '''SELECT Samples.Name, Samples.SampleID, XData, YData
                FROM Samples, Spectra WHERE Samples.SampleID = Spectra.SampleID
                AND Spectra.SpectrumID = ?'''
        result = self.cursor.execute(query, (spectrumID,))
        results = result.fetchall()
        if len(results) < 1:
            raise "Measurement record not found"

        sig = Signature()
        sig.measurement_id = spectrumID
        sig.sample_name = results[0][0]
        sig.sample_id = results[0][1]
        x = array.array(arraytypecode)
        x.fromstring(results[0][2])
        sig.x = list(x)
        y = array.array(arraytypecode)
        y.fromstring(results[0][3])
        sig.y = list(y)
        return sig

    def query(self, sql, args=None):
        '''Returns the result of an arbitrary SQL statement.

        Arguments:

            `sql` (str):

                An SQL statement to be passed to the database. Use "?" for
                variables passed into the statement.

            `args` (tuple):

                Optional arguments which will replace the "?" placeholders in
                the `sql` argument.

        Returns:

            An :class:`sqlite3.Cursor` object with the query results.

        Example::

            >>> sql = r'SELECT SpectrumID, Name FROM Samples, Spectra ' +
            ...        'WHERE Spectra.SampleID = Samples.SampleID ' +
            ...        'AND Name LIKE "%grass%" AND MinWavelength < ?'
            >>> args = (0.5,)
            >>> cur = db.query(sql, args)
            >>> for row in cur:
            ...     print row
            ...
            (356, u'dry grass')
            (357, u'grass')
        '''
        if args:
            return self.cursor.execute(sql, args)
        else:
            return self.cursor.execute(sql)

    def print_query(self, sql, args=None):
        '''Prints the text result of an arbitrary SQL statement.

        Arguments:

            `sql` (str):

                An SQL statement to be passed to the database. Use "?" for
                variables passed into the statement.

            `args` (tuple):

                Optional arguments which will replace the "?" placeholders in
                the `sql` argument.

        This function performs the same query function as
        :meth:`spectral.database.Asterdatabase.query` except query results are
        printed to **stdout** instead of returning a cursor object.

        Example:

            >>> sql = r'SELECT SpectrumID, Name FROM Samples, Spectra ' +
            ...        'WHERE Spectra.SampleID = Samples.SampleID ' +
            ...        'AND Name LIKE "%grass%" AND MinWavelength < ?'
            >>> args = (0.5,)
            >>> db.print_query(sql, args)
            356|dry grass
            357|grass
        '''
        ret = self.query(sql, args)
        for row in ret:
            print("|".join([str(x) for x in row]))

    def create_envi_spectral_library(self, spectrumIDs, bandInfo):
        '''Creates an ENVI-formatted spectral library for a list of spectra.

        Arguments:

            `spectrumIDs` (list of ints):

                List of **SpectrumID** values for of spectra in the "Spectra"
                table of the ASTER database.

            `bandInfo` (:class:`~spectral.BandInfo`):

                The spectral bands to which the original ASTER library spectra
                will be resampled.

        Returns:

            A :class:`~spectral.io.envi.SpectralLibrary` object.

        The IDs passed to the method should correspond to the SpectrumID field
        of the ASTER database "Spectra" table.  All specified spectra will be
        resampled to the same discretization specified by the bandInfo
        parameter. See :class:`spectral.BandResampler` for details on the
        resampling method used.
        '''
        from spectral.algorithms.resampling import BandResampler
        from spectral.io.envi import SpectralLibrary
        import numpy
        import unicodedata
        spectra = numpy.empty((len(spectrumIDs), len(bandInfo.centers)))
        names = []
        for i in range(len(spectrumIDs)):
            sig = self.get_signature(spectrumIDs[i])
            resample = BandResampler(
                sig.x, bandInfo.centers, None, bandInfo.bandwidths)
            spectra[i] = resample(sig.y)
            names.append(unicodedata.normalize('NFKD', sig.sample_name).
                         encode('ascii', 'ignore'))
        header = {}
        header['wavelength units'] = 'um'
        header['spectra names'] = names
        header['wavelength'] = bandInfo.centers
        header['fwhm'] = bandInfo.bandwidths
        return SpectralLibrary(spectra, header, {})
