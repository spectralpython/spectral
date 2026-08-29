'''
Code for reading and managing relab spectral library data.
'''

from .spectral_database import SpectralDatabase

readline = lambda fin: fin.readline()
open_file = lambda filename: open(filename, encoding='iso-8859-1')

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
    '''Object to store sample/measurement metadata, as well as wavelength-signature vectors.'''
    def __init__(self):
        self.sample = {}
        self.measurement = {}


def read_relab_file(filename):
    '''Reads a relab spectrum file.
    .asc files are structured as:
      Number of data lines
      Data lines (Wavelength in nm, Reflectance, and SD if any)
      Three blank lines
      File name
      A blank line
      Sample ID
      Comment lines

    Note: Not considering any Standard Deviation (SD) data.
    '''
    with open_file(filename) as fin:
        lines = [line.rstrip('\n') for line in fin]

    number_of_data = int(lines[0])

    x = []
    y = []
    for i in range(1, 1+number_of_data):
        current_line = lines[i]
        parse1 = current_line.strip().split()
        x_val = float(parse1[0])
        y_val = float(parse1[1])
        x.append(x_val)
        y.append(y_val)

    # Make sure wavelengths are ascending
    if float(x[0]) > float(x[-1]):
        x.reverse()
        y.reverse()

    s = Signature()

    s.x = x
    s.y = y
    s.measurement['first x value'] = x[0]
    s.measurement['last x value'] = x[-1]
    s.measurement['number of x values'] = number_of_data

    # Extract ReLab ID and store it
    relab_id = lines[number_of_data+6].strip()  # a string
    s.sample["relab_id"] = relab_id
    s.measurement["relab_id"] = relab_id

    # ideally can be taken from argument to this function as well
    # converting to lower because actual names are in lowercase but stored as
    # uppercase in files
    filename_from_content = lines[number_of_data+4
                                  ].strip().replace(' ', '').lower()

    s.sample['name'] = filename_from_content

    # The parsing of comment lines
    comment_lines = lines[number_of_data+7:]

    if len(comment_lines) >= 2:  # what if there are no commont lines
        last_line = comment_lines[-1]
        last_but_one_line = comment_lines[-2]
    else:
        last_line = ''
        last_but_one_line = ''

    if last_line.strip() and last_but_one_line.strip():  # both non black lines
        if last_line.strip().split()[0] == 'Date:' and  \
                last_but_one_line.strip().split()[0] == 'Source':
            # all these modifications are to get rid of edge cases
            modified_line = last_but_one_line.strip().replace(':', ' ')
            modified_line = modified_line.replace(',', ' ')
            split = modified_line.split()
            if len(split) == 8:
                s.measurement['source_angle'] = float(split[2])
                s.measurement['detect_angle'] = float(split[5])
                s.measurement['volt'] = float(split[7])
            modified_line = last_line.strip().replace(',', ' ')
            split = modified_line.split()
            if len(split) == 4:
                s.sample['date'] = split[1]
                s.sample['time'] = split[3]

            # delete the last two lines now
            comment_lines = comment_lines[:-2]

    description = relab_id
    for i in comment_lines:
        description += ' ' + i.strip() if i.strip() else ''

    s.sample['description'] = description

    return s


class RelabDatabase(SpectralDatabase):
    '''A relational database to manage relab spectral library data.'''
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
        xBlob = sqlite3.Binary(array.array(arraytypecode, xData).tobytes())
        yBlob = sqlite3.Binary(array.array(arraytypecode, yData).tobytes())
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
    def create(cls, filename, relab_data_dir=None):
        '''Creates an relab relational database by parsing RELAB data files.

        Arguments:

            `filename` (str):

                Name of the new sqlite database file to create.

            `relab_data_dir` (str):

                Path to the directory containing relab library data files. If
                this argument is not provided, no data will be imported.

        Returns:

            An :class:`~spectral.database.RelabDatabase` object.

        Example::

            >>> RelabDatabase.create("relab_lib.db", "/STORAGE/ReLab/data")

        This is a class method (it does not require instantiating an
        RelabDatabase object) that creates a new database by parsing all of the
        files in the relab library data directory.  Normally, this should only
        need to be called once.  Subsequently, a corresponding database object
        can be created by instantiating a new RelabDatabase object with the
        path the database file as its argument.  For example::

            >>> from spectral.database.relab import RelabDatabase
            >>> db = RelabDatabase("relab_lib.db")
        '''
        import os
        if os.path.isfile(filename):
            raise Exception('Error: Specified file already exists.')
        db = cls()
        db._connect(filename)
        for schema in cls.schemas:
            db.cursor.execute(schema)
        if relab_data_dir:
            db._import_files(relab_data_dir)
        return db

    def __init__(self, sqlite_filename=None):
        '''Creates a database object to interface an existing database.

        Arguments:

            `sqlite_filename` (str):

                Name of the database file.  If this argument is not provided,
                an interface to a database file will not be established.

        Returns:

            An :class:`~spectral.RelabDatabase` connected to the database.
        '''
        from spectral.io.spyfile import find_file_path
        if sqlite_filename:
            self._connect(find_file_path(sqlite_filename))
        else:
            self.db = None
            self.cursor = None

    def read_file(self, filename):
        return read_relab_file(filename)

    def _import_files(self, data_dir, ignore=bad_files):
        '''Read each file in the relab library and convert to AVIRIS bands.'''
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
        
        # Get all .asc files in subsubdirs 
        files = glob(data_dir+'/**/*/*.asc', recursive=True)
        print(len(files))

        for f in range(len(files)):
            print('Importing %s.' % files[f])
            try:
                sig = self.read_file(files[f])
                s = sig.sample
                sampleNum = s['relab_id']
            except:
                raise Exception ('Error creating SIG' % (files[f]))

            phase = 'solid'
            s['type'] = "manmade_natural"
            s['class'] = " "
            s['subclass'] = " "
            s['particle size'] = " "
            s['owner'] = " "
            s['origin'] = "RELab"
            try:
                idd = self._add_sample(s['name'], s['type'], s['class'], s['subclass'], s['particle size'],
                        sampleNum, s['owner'], s['origin'], phase, s['description'])
            except:
                raise Exception ('Error creating IDD')

            instrument = os.path.basename(files[f]).split('.')[0]
            environment = 'lab'
            m = sig.measurement

            # Correct numerous mispellings of "reflectance" and "transmittance"
            m['y units'] = 'reflectance (percent)'
            yUnit = m['y units']
            measurement = 'reflectance'
            m['x units'] = 'wavelength (nm)'
            try:
                self._add_signature(idd, -1, instrument, environment, measurement,
                                m['x units'], yUnit, m['first x value'],
                                m['last x value'], sig.x, sig.y)
            except:
                raise Exception ('Error creating Signature')

        if numFiles == 0:
            print('No data files were found in directory "%s".' % data_dir)
        else:
            print('Processed %d files.' % numFiles)
        if numIgnored > 0:
            print('Ignored the following %d bad files:' % (numIgnored))
            for f in filesToIgnore:
                print('\t' + f)

        return sigs

    def get_spectrum(self, spectrumID):
        '''Returns a spectrum from the database.

        Usage:

            (x, y) = relab.get_spectrum(spectrumID)

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
            raise Exception('Measurement record not found')
        x = array.array(arraytypecode)
        x.frombytes(rows[0][0])
        y = array.array(arraytypecode)
        y.frombytes(rows[0][1])
        return (list(x), list(y))

    def get_signature(self, spectrumID):
        '''Returns a spectrum with some additional metadata.

        Usage::

            sig = relab.get_signature(spectrumID)

        Arguments:

            `spectrumID` (int):

                The **SpectrumID** value for the desired spectrum from the
                **Spectra** table in the database.

        Returns:

            `sig` (:class:`~spectral.database.relab.Signature`):

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
            raise Exception('Measurement record not found')

        sig = Signature()
        sig.measurement_id = spectrumID
        sig.sample_name = results[0][0]
        sig.sample_id = results[0][1]
        x = array.array(arraytypecode)
        x.frombytes(results[0][2])
        sig.x = list(x)
        y = array.array(arraytypecode)
        y.frombytes(results[0][3])
        sig.y = list(y)
        return sig

    def create_envi_spectral_library(self, spectrumIDs, bandInfo):
        '''Creates an ENVI-formatted spectral library for a list of spectra.

        Arguments:

            `spectrumIDs` (list of ints):

                List of **SpectrumID** values for of spectra in the "Spectra"
                table of the relab database.

            `bandInfo` (:class:`~spectral.BandInfo`):

                The spectral bands to which the original relab library spectra
                will be resampled.

        Returns:

            A :class:`~spectral.io.envi.SpectralLibrary` object.

        The IDs passed to the method should correspond to the SpectrumID field
        of the relab database "Spectra" table.  All specified spectra will be
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
