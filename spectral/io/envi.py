#########################################################################
#
#   envi.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2001-2010 Thomas Boggs
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
Code for creating SpyFile objects from files with ENVI header files.
'''

def readEnviHdr(file):
    '''
    USAGE: hdr = readEnviHeader(file)

    Reads an ENVI ".hdr" file header and returns the parameters in
    a dictionary as strings.
    '''

    from string import find, split, strip
    from exceptions import IOError
    from __builtin__ import open
    
    f = open(file, 'r')
    
    if find(f.readline(), "ENVI") == -1:
        f.close()
        raise IOError, "Not an ENVI header."

    lines = f.readlines()
    f.close()

    dict = {}
    i = 0
    try:
        while i < len(lines):
            if find(lines[i], '=') == -1:
                i += 1
                continue
            (key, val) = split(lines[i], '=')
            key = strip(key)
            val = strip(val[:-1])
            if val[0] == '{':
                str = val
                while str[-1] != '}':
                    i += 1
                    str += strip(lines[i][:-1])
                
                if key == 'description':
                    dict[key] = str[1:-1]
                else:
                    vals = split(str[1:-1], ',')
                    for j in range(len(vals)):
                        vals[j] = strip(vals[j])
                    dict[key] = vals
            else:
                dict[key] = val
            i += 1
        return dict
    except:
        raise IOError, "Error while reading ENVI file header."

def open(file, image = None):
    '''Creates a SpyFile object for a file with and ENVI HDR header file.'''

    import os
    from exceptions import IOError, TypeError
    from spyfile import findFilePath
    import numpy
    import spectral

    headerPath = findFilePath(file)
    h = readEnviHdr(headerPath)
    h["header file"] = file

    class Params: pass
    p = Params()
    p.nBands = int(h["bands"])
    p.nRows = int(h["lines"])
    p.nCols = int(h["samples"])
    p.offset = int(h["header offset"])
    p.byteOrder = int(h["byte order"])

    #  Validate image file name
    if not image:
        #  Try to determine the name of the image file
        headerDir = os.path.split(headerPath)
        if headerPath[-4:].lower() == '.hdr':
            headerPathTitle = headerPath[:-4]
            exts = ['', '.img', '.IMG', '.dat', '.DAT', '.sli', '.SLI']
            for ext in exts:
                testname = headerPathTitle + ext
                if os.path.isfile(testname):
                    image = testname
                    break
        if not image:
            raise IOError, 'Unable to determine image file name.'
    p.fileName = image

    #  Determine numeric data type
    if h["data type"] == '1':
        # byte
        p.format = 'b'
        p.typecode = 'b'
    elif h["data type"] == '2':
        # 16-bit int
        p.format = 'h'
        p.typecode = 'h'
    elif h["data type"] == '3':
        # 32-bit int
        p.format = 'f'
        p.typecode = 'f'
    elif h["data type"] == '4':
        #  32-bit float
        p.format = 'f'
        p.typecode = 'f'
    elif h["data type"] == '5':
        #  64-bit float
        p.format = 'd'
        p.typecode = 'd'
    elif h["data type"] == '6':
        #  2x32-bit complex
        p.format = 'F'
        p.typecode = 'F'
    elif h["data type"] == '9':
        #  2x64-bit complex
        p.format = 'D'
        p.typecode = 'D'
    elif h["data type"] == '12':
        #  16-bit unsigned int
        p.format = 'H'
        p.typecode = 'H'
    elif h["data type"] == '13':
        #  32-bit unsigned int
        p.format = 'I'
        p.typecode = 'I'
    elif h["data type"] == '14':
        #  64-bit int
        p.format = 'q'
        p.typecode = 'q'
    elif h["data type"] == '15':
        #  64-bit unsigned int
        p.format = 'Q'
        p.typecode = 'Q'
    else:
        #  Don't recognize this type code
        raise TypeError, 'Unrecognized data type code in header ' + \
              'file.  If you believe the header to be correct, please' + \
              'submit a bug report to have the type code added.'

    if h.get('file type') == 'ENVI Spectral Library':
	# File is a spectral library
	data = numpy.fromfile(p.fileName, p.format, p.nCols * p.nRows)
	data.shape = (p.nRows, p.nCols)
	if (p.byteOrder != spectral.byteOrder):
	    data = data.byteswap()
	return SpectralLibrary(data, h, p)
    
    #  Create the appropriate object type for the interleave format.
    inter = h["interleave"]
    if inter == 'bil' or inter == 'BIL':
        from spectral.io.bilFile import BilFile
        img = BilFile(p, h)
    elif inter == 'bip' or inter == 'BIP':
        from spectral.io.bipfile import BipFile
        img = BipFile(p, h)
    else:
        from spectral.io.bsqfile import BsqFile
        img = BsqFile(p, h)
    
    img.scaleFactor = float(h.get('reflectance scale factor', 1.0))
    
    # Add band info
    
    if h.has_key('wavelength'):
	try:
	    img.bands.centers = [float(b) for b in h['wavelength']]
	except:
	    pass
    if h.has_key('fwhm'):
	try:
	    img.bands.bandwidths = [float(f) for f in h['fwhm']]
	except:
	    pass
    img.bands.bandUnit = h.get('wavelength units', "")
    img.bands.bandQuantity = "Wavelength"
    
    return img

class SpectralLibrary:
    '''
    The envi.SpectralLibrary class holds data contained in an ENVI-formatted spectral
    library file (.sli files), which stores data as specified by a corresponding .hdr
    header file.  The primary members of an Envi.SpectralLibrary object are:
    
	spectra			A subscriptable array of all spectra in the library.
	names			A list of names corresponding to the spectra.
	bands			A BandInfo object defining associated spectral bands.
	
    '''
    
    def __init__(self, data, header, params):
	from spectral.spectral import BandInfo
	self.spectra = data
	self.bands = BandInfo()
	if header.has_key('wavelength'):
	    try:
		self.bands.centers = [float(b) for b in header['wavelength']]
	    except:
		pass
	if header.has_key('fwhm'):
	    try:
		self.bands.bandwidths = [float(f) for f in header['fwhm']]
	    except:
		pass
	if header.has_key('spectra names'):
	    self.names = header['spectra names']
	else:
	    self.names = [''] * self.bands.shape[0]
	self.bands.bandUnit = header.get('wavelength units', "")
	self.bands.bandQuantity = "Wavelength"
	self.params = params
	self.metadata = {}
	self.metadata.update(header)
	self.metadata['data ignore value'] = 'NaN'
	
    def save(self, fileBaseName, description = None):
	import spectral
	import __builtin__
	meta = {}
	meta.update(self.metadata)
	if self.bands.centers:
	    meta['samples'] = len(self.bands.centers)
	else:
	    meta['samples'] = len(self.spectra.shape[0])
	meta['lines'] = self.spectra.shape[0]
	meta['bands'] = 1
	meta['header offset'] = 0
	meta['data type'] = 4		# 32-bit float
	meta['interleave'] = 'bsq'
	meta['byte order'] = spectral.byteOrder
	meta['wavelength units'] = self.bands.bandUnit
	meta['spectra names'] = [str(n) for n in self.names]
	meta['wavelength'] = self.bands.centers
	meta['fwhm'] = self.bands.bandwidths
	if (description):
	    meta['description'] = description
	_writeEnviHdr(fileBaseName + '.hdr', meta, True)
	fout = __builtin__.open(fileBaseName + '.sli', 'wb')
	self.spectra.astype('f').tofile(fout)
	fout.close()

def _writeHeaderParam(fout, paramName, paramVal):
    if not isinstance(paramVal, str) and hasattr(paramVal, '__len__'):
	valStr = '{ %s }' % (' , '.join([str(v) for v in paramVal]),)
    else:
	valStr = str(paramVal)
    fout.write('%s = %s\n' % (paramName, valStr))
	
def writeEnviHdr(fileName, headerDict, isLibrary = False):
    import __builtin__
    fout = __builtin__.open(fileName, 'w')
    d = {}
    d.update(headerDict)
    d['file type'] = 'ENVI Spectral Library'
    fout.write('ENVI\n')
    for k in d:
	writeHeaderParam(fout, k, d[k])
    fout.close()
    
