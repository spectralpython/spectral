#########################################################################
#
#   Aviris.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2001-2008 Thomas Boggs
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

'''
Functions for handling AVIRIS image files.
'''

def open(file, bandFile = None):
    '''
    Creates a SpyFile object for an AVIRIS image file.
    
    USAGE: img = Aviris(fileName [, bandFile])
    
    ARGS:
        fileName		Name of the data cube file
	bandFile		Name of the AVIRIS spectral calibration file
    '''

    from Spectral.Io.BipFile import BipFile
    import os, glob
    from exceptions import IOError
    from SpyFile import findFilePath

    class Params: pass
    p = Params()

    p.fileName = findFilePath(file)
    p.nBands = 224
    p.nCols = 614
    fileSize = os.stat(p.fileName)[6]
    if fileSize % 275072 != 0:
        raise IOError, 'File size not consitent with Aviris format.'
    p.nRows = int(fileSize / 275072)
    p.format = 'h'
    p.typecode = 'h'
    p.offset = 0
    p.byteOrder = 1
    metadata = {'default bands' : ['29', '18', '8']}

    img = BipFile(p, metadata)
    img.scaleFactor = 10000.0
    
    if bandFile:
	img.bands = readAvirisBands(findFilePath(bandFile))
    else:
	# Let user know if band cal files are available
	fileDir = os.path.split(p.fileName)[0]
	calFiles = glob.glob(fileDir + '/*.spc')
	if len(calFiles) > 0:
	    print '\nThe following band calibration files are located in the same ' \
	          'directory as the opened AVIRIS file:\n'
	    for f in calFiles:
		print "    " + os.path.split(f)[1]
	    print '\nTo associate a band calibration file with an AVIRIS data file, ' \
	          're-open the AVIRIS file with the following syntax:\n'
	    print '    >>> img = openAviris(fileName, calFileName)\n'
    return img

def readAvirisBands(calFileName):
    '''
    Returns a pair of lists containing the center wavelengths and full widths
    at half maximum (fwhm) for all AVIRIS bands, in microns (um).
    '''
    import __builtin__
    from Spectral import BandInfo
    bands = BandInfo()
    bands.bandQuantity = 'Wavelength'
    bands.bandUnit = 'nm'
    
    fin = __builtin__.open(calFileName)
    rows = [line.split() for line in fin]
    rows = [[float(x) for x in row] for row in rows if len(row) == 5]
    columns = zip(*rows)
    bands.centers = columns[0]
    bands.bandwidths = columns[1]
    bands.centersStdDevs = columns[2]
    bands.bandwidthsStdDevs = columns[3]
    return bands
