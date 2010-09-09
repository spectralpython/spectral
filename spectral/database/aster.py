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


tableSchemas = [
    'CREATE TABLE Samples (SampleID INTEGER PRIMARY KEY, Name TEXT, Type TEXT, Class TEXT, SubClass TEXT, ' \
			  'ParticleSize TEXT, SampleNum TEXT, Owner TEXT, Origin TEXT, Phase TEXT, Description TEXT)',
    'CREATE TABLE Spectra (SpectrumID INTEGER PRIMARY KEY, SampleID INTEGER, SensorCalibrationID INTEGER, ' \
			  'Instrument TEXT, Environment TEXT, Measurement TEXT, ' \
			  'XUnit TEXT, YUnit TEXT, MinWavelength FLOAT, MaxWavelength FLOAT, ' \
			  'NumValues INTEGER, XData BLOB, YData BLOB)',
    ]

arrayTypeCode = 'f'

# These files contained malformed signature data and will be ignored.
badFiles = [
    'jhu.nicolet.mineral.silicate.tectosilicate.fine.albite1.spectrum.txt',
    'usgs.perknic.rock.igneous.mafic.colid.me3.spectrum.txt'
    ]


def readPair(fin, numLines = 1):
    '''Reads a colon-delimited attribute-value pair from the file stream.'''
    s = ''
    for i in range(numLines):
        s += " " + fin.readline().strip()
    return [x.strip().decode('iso-8859-1').lower() for x in s.split(':')]
    
class Signature:
    '''Object to store sample/measurement metadata, as well as wavelength-signatrure vectors.'''
    def __init__(self):
        self.sample = {}
        self.measurement = {}

def readFile(fileName):
    '''Reads an ASTER 2.x spectrum file.'''
    fin = open(fileName)
    s = Signature()

    # Number of lines per metadata attribute value
    lpv = [1] * 8 + [2] + [6]
    
    # A few files have an additional "Colleted by" sample metadata field, which
    # sometimes affects the number of header lines
    
    haveCollectedBy = False
    for i in range(30):
        line = fin.readline().strip()
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
        pair = readPair(fin, lpv[i])
        s.sample[pair[0].lower()] = pair[1]

    # Read measurement metadata
    lpv = [1] * 8 + [2]
    for i in range(len(lpv)):
        pair = readPair(fin, lpv[i])
        if len(pair) < 2:
            print pair
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
#	    print 'single item (%s) on signature line, %s' %  (pair[0], fileName)
	    continue
	elif nItems > 2:
	    print 'more than 2 values on signature line,', fileName
	    continue
	try:
	    x = float(pair[0])
	except:
	    print 'corrupt signature line,', fileName
	if x == 0:
#	    print 'Zero wavelength value', fileName
	    continue
	elif x < 0:
	    print 'Negative wavelength value,', fileName
	    continue

        pairs.append(pair)
    
    [x, y]  = [list(v) for v in zip(*pairs)]

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
    schemas = tableSchemas
    
    def _addSample(self, name, sampleType, sampleClass, subClass, particleSize, sampleNumber, owner, origin, phase, description):
	sql = '''INSERT INTO Samples (Name, Type, Class, SubClass, ParticleSize, SampleNum, Owner, Origin, Phase, Description)
		    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
	self.cursor.execute(sql, (name, sampleType, sampleClass, subClass, particleSize, sampleNumber, owner, origin, phase, description))
	rowId =  self.cursor.lastrowid
	self.db.commit()
	return rowId
    
    def _addSignature(self, sampleID, calibrationID, instrument, environment, measurement,
		     xUnit, yUnit, minWavelength, maxWavelength, xData, yData):
	import sqlite3
	import array
	sql = '''INSERT INTO Spectra (SampleID, SensorCalibrationID, Instrument,
		 Environment, Measurement, XUnit, YUnit, MinWavelength, MaxWavelength,
		 NumValues, XData, YData) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
	xBlob = sqlite3.Binary(array.array(arrayTypeCode, xData).tostring())
	yBlob = sqlite3.Binary(array.array(arrayTypeCode, yData).tostring())
	numValues = len(xData)
	self.cursor.execute(sql, (sampleID, calibrationID, instrument, environment, measurement, \
			    xUnit, yUnit, minWavelength, maxWavelength, numValues, xBlob, yBlob))
	rowId =  self.cursor.lastrowid
	self.db.commit()
	return rowId
	
    @classmethod
    def create(cls, fileName, asterDataDir = None):
	'''Creates and ASTER relational database by parsing ASTER data files.
	
	USAGE:
	    db = AsterDatabase.create(fileName, asterDataDir)
	ARGS:
	    fileName		Name of the new sqlite database file to create.
	    asterDataDir	Path to the directory containing ASTER data files.
	RETURN:
	    db			An AsterDatabase object for the new database.
	EXAMPLE:
	    >>> from spectral.database.aster import AsterDatabase
	    >>> AsterDatabase.create("aster_lib.db", "/CDROM/ASTER2.0/data")
	    
	This is a class method (it does not require instantiating an AsterDatabase
	object) that creates a new database by parsing all of the files in the
	ASTER library data directory.  Normally, this should only need to be called
	once.  Subsequently, a corresponding database object can be created by
	instantiating a new AsterDatabase object with the path the database file
	as its argument.  For example:
	
	    >>> from spectral.database.aster import AsterDatabase
	    >>> db = AsterDatabase("aster_lib.db")
	'''
        import os
        if os.path.isfile(fileName):
            raise Exception('Error: Specified file already exists.')
	db = AsterDatabase(fileName)
	for schema in cls.schemas:
	    db.cursor.execute(schema)
        if asterDataDir:
            db._importAsterFiles(asterDataDir)
	return db
	
    def __init__(self, sqliteFileName = None):
	if sqliteFileName:
	    self._connect(sqliteFileName)
	else:
	    self.db = None
	    self.cursor = None
        
    def _importAsterFiles(self, asterDataDir):
        '''Read each file in the ASTER library and convert to AVIRIS bands.'''
        from glob import glob
        import numpy
        import os
        
        if not os.path.isdir(asterDataDir):
            raise Exception('Error: Invalid directory name specified.')
        filesToIgnore = [asterDataDir + '/' + f for f in badFiles]
    
        numFiles = 0
        numIgnored = 0
    
        sigID = 1
        class Sig:
            pass
        sigs = []
        
        for f in glob(asterDataDir + '/*spectrum.txt'):
            if f in filesToIgnore:
                numIgnored += 1
                continue
            print('Importing %s.' % f)
            numFiles += 1
            sig = readFile(f)
            s = sig.sample
            if s['particle size'].lower == 'liquid':
                phase = 'liquid'
            else:
                phase = 'solid'
            if s.has_key('sample no.'):
                sampleNum = s['sample no.']
            else:
                sampleNum = ''
            id = self._addSample(s['name'], s['type'], s['class'], s['subclass'], s['particle size'],
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
            self._addSignature(id, -1, instrument, environment, measurement,
                               m['x units'], yUnit, m['first x value'], m['last x value'], sig.x, sig.y)
        if numFiles == 0:
            print 'No ASTER data files were found in directory "%s".' % asterDataDir
        else:
            print 'Processed %d files.' % numFiles
        if numIgnored > 0:
            print 'Ignored the following %d bad files:' % (numIgnored)
            for f in filesToIgnore:
                print '\t' + f

        return sigs
	
    def _connect(self, sqliteFileName):
	'''Establishes a connection to the Specbase sqlite database.'''
	import sqlite3
	self.db = sqlite3.connect(sqliteFileName)
	self.cursor = self.db.cursor()
	
    def getSpectrum(self, spectrumID):
	'''Returns a spectrum from the database.
	
	USAGE:
	    (x, y) = aster.getSpectrum(spectrumID)
	ARGS:
	    spectrumID		SpectrumID from the Spectra database table
	RETURN:
	    x			Band centers for the spectrum
	    y			Spectrum data values
	    
	Returns a pair of vectors containing the wavelengths and measured values
	values of a measurment.  For additional metadata, call "getSignature"
	instead.
	'''
	import array
	query = '''SELECT XData, YData FROM Spectra WHERE SpectrumID = ?'''
	result =  self.cursor.execute(query, (spectrumID,))
	rows = result.fetchall()
	if len(rows) < 1:
	    raise 'Measurement record not found'
	x = array.array(arrayTypeCode)
	x.fromstring(rows[0][0])
	y = array.array(arrayTypeCode)
	y.fromstring(rows[0][1])
	return (list(x), list(y))

    def getSignature(self, spectrumID):
	'''
	USAGE:
	    sig = aster.getSignature(spectrumID)
	ARGS:
	    spectrumID		SpectrumID from the Spectra database table
	RETURN:
	    sig			An object with the following attributes:
				    - measurementID (from Spectra table)
				    - sampleName
				    - sampleID
				    - x (list of band center wavelengths)
				    - y (list of spectrum values)
	    
	Returns an object containing the wavelengths, measured values, sampleID,
	and sample name.
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
	sig.measurementID = spectrumID
	sig.sampleName = results[0][0]
	sig.sampleID = results[0][1]
	x = array.array(arrayTypeCode)
	x.fromstring(results[0][2])
        sig.x = list(x)
	y = array.array(arrayTypeCode)
	y.fromstring(results[0][3])
        sig.y = list(y)
	return sig
    
    def query(self, sqlQuery):
	'''Returns the text result of an arbitrary SQL statement.'''
	ret = self.cursor.execute(sqlQuery)
	lines = ["|".join([str(x) for x in row ]) for row in ret]
	return '\n'.join(lines)
    
    def createEnviSpectralLibrary(self, spectrumIDs, bandInfo):
	'''Creates an ENVI-formatted spectral library for a list of spectra.
	
	USAGE:
	    lib = aster.createEnviSpectralLibrary(spectrumIDs, bandInfo)
	ARGS:
	    spectrumIDs		A list of IDs of spectra in the "Spectra" table
				of the ASTER database.
	    bandInfo		A BandInfo object specifying the spectral bands
				to which the original spectra will be resampled.
	
	This method returns a SpectralLibrary object as defined in the
	spectral.io.envi module.  The IDs passed to the method should correspond
	to the SpectrumID field of the ASTER database "Spectra" table.  All
	specified spectra will be resampled to the same discretization specified
	by the bandInfo parameter.  See BandResampler for details on the
	resampling method used.				
	'''
        from spectral.algorithms.resampling import BandResampler
	from spectral.io.envi import SpectralLibrary
        import numpy
        spectra = numpy.empty((len(spectrumIDs), len(bandInfo.centers)))
        names = []
        for i in range(len(spectrumIDs)):
            sig = self.getSignature(spectrumIDs[i])
            resample = BandResampler(sig.x, bandInfo.centers, None, bandInfo.bandwidths)
            spectra[i] = resample(sig.y)
            names.append(sig.sampleName)
	header = {}
	header['wavelength units'] = 'um'
	header['spectra names'] = names
	header['wavelength'] = bandInfo.centers
	header['fwhm'] = bandInfo.bandwidths
	return SpectralLibrary(spectra, header, {})

