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
ENVI [#envi-trademark]_ is a popular commercial software package for processing and analyzing
geospatial imagery.  SPy supports reading imagery with associated ENVI header files
and reading & writing spectral libraries with ENVI headers.  ENVI files are opened
automatically by the SPy :func:`~spectral.image` function but can also be called
explicitly.  It may be necessary to open an ENVI file explicitly if the data file
is in a separate directory from the header or if the data file has an unusual file
extension that SPy can not identify.

    >>> import spectral.io.envi as envi
    >>> img = envi.open('cup95eff.int.hdr', '/Users/thomas/spectral_data/cup95eff.int')

.. [#envi-trademark] ENVI is a registered trademark of ITT Corporation.
'''

def read_envi_header(file):
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
	    (key, sep, val) = lines[i].partition("=")
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
    '''
    Opens an image or spectral library with an associated ENVI HDR header file.

    Arguments:
    
	`file` (str):
	
	    Name of the header file for the image.
	
	`image` (str):
	
	    Optional name of the associated image data file.
	
    Returns:
	:class:`spectral.SpyFile` or :class:`spectral.io.envi.SpectralLibrary` object.
    
    Raises:
	TypeError, IOError.
	
    If the specified file is not found in the current directory, all directories
    listed in the SPECTRAL_DATA environment variable will be searched until the
    file is found.  Based on the name of the header file, this function will
    search for the image file in the same directory as the header, looking for a
    file with the same name as the header but different extension. Extensions
    recognized are .img, .dat, .sli, and no extension.  Capitalized versions of
    the file extensions are also searched.
    '''

    import os
    from exceptions import IOError, TypeError
    from spyfile import find_file_path
    import numpy
    import spectral

    headerPath = find_file_path(file)
    h = read_envi_header(headerPath)
    h["header file"] = file

    class Params: pass
    p = Params()
    p.nbands = int(h["bands"])
    p.nrows = int(h["lines"])
    p.ncols = int(h["samples"])
    p.offset = int(h["header offset"])
    p.byte_order = int(h["byte order"])

    inter = h["interleave"]

    #  Validate image file name
    if not image:
        #  Try to determine the name of the image file
        headerDir = os.path.split(headerPath)
        if headerPath[-4:].lower() == '.hdr':
            headerPathTitle = headerPath[:-4]
            exts = ['', 'img', 'IMG', 'dat', 'DAT', 'sli', 'SLI'] + [inter.lower(), inter.upper()]
            for ext in exts:
		if len(ext) == 0:
		    testname = headerPathTitle
		else:
		    testname = headerPathTitle + '.' + ext
                if os.path.isfile(testname):
                    image = testname
                    break
        if not image:
            raise IOError, 'Unable to determine image file name.'
    p.filename = image

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
	data = numpy.fromfile(p.filename, p.format, p.ncols * p.nrows)
	data.shape = (p.nrows, p.ncols)
	if (p.byte_order != spectral.byte_order):
	    data = data.byteswap()
	return SpectralLibrary(data, h, p)
    
    #  Create the appropriate object type for the interleave format.
    inter = h["interleave"]
    if inter == 'bil' or inter == 'BIL':
        from spectral.io.bilfile import BilFile
        img = BilFile(p, h)
    elif inter == 'bip' or inter == 'BIP':
        from spectral.io.bipfile import BipFile
        img = BipFile(p, h)
    else:
        from spectral.io.bsqfile import BsqFile
        img = BsqFile(p, h)
    
    img.scale_factor = float(h.get('reflectance scale factor', 1.0))
    
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
    img.bands.band_unit = h.get('wavelength units', "")
    img.bands.bandQuantity = "Wavelength"
    
    return img

class SpectralLibrary:
    '''
    The envi.SpectralLibrary class holds data contained in an ENVI-formatted spectral
    library file (.sli files), which stores data as specified by a corresponding .hdr
    header file.  The primary members of an Envi.SpectralLibrary object are:
    
	`spectra` (:class:`numpy.ndarray`):
	
	    A subscriptable array of all spectra in the library. `spectra` will
	    have shape `CxB`, where `C` is the number of spectra in the library
	    and `B` is the number of bands for each spectrum.
	
	`names` (list of str):
	
	    A length-`C` list of names corresponding to the spectra.
	    
	`bands` (:class:`spectral.BandInfo`):
	
	    Spectral bands associated with the library spectra.
	
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
	self.bands.band_unit = header.get('wavelength units', "")
	self.bands.bandQuantity = "Wavelength"
	self.params = params
	self.metadata = {}
	self.metadata.update(header)
	self.metadata['data ignore value'] = 'NaN'
	
    def save(self, fileBaseName, description = None):
	'''
	Saves the spectral library to a library file.

	Arguments:
	
	    `fileBaseName` (str):
	    
		Name of the file (without extension) to save.
	    
	    `description` (str):
	    
		Optional text description of the library.

	This method creates two files: `fileBaseName`.hdr and `fileBaseName`.sli.
	'''
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
	meta['byte order'] = spectral.byte_order
	meta['wavelength units'] = self.bands.band_unit
	meta['spectra names'] = [str(n) for n in self.names]
	meta['wavelength'] = self.bands.centers
	meta['fwhm'] = self.bands.bandwidths
	if (description):
	    meta['description'] = description
	write_envi_header(fileBaseName + '.hdr', meta, True)
	fout = __builtin__.open(fileBaseName + '.sli', 'wb')
	self.spectra.astype('f').tofile(fout)
	fout.close()

def _write_header_param(fout, paramName, paramVal):
    if not isinstance(paramVal, str) and hasattr(paramVal, '__len__'):
	valStr = '{ %s }' % (' , '.join([str(v).replace(',', '-') for v in paramVal]),)
    else:
	valStr = str(paramVal)
    fout.write('%s = %s\n' % (paramName, valStr))
	
def write_envi_header(fileName, header_dict, is_library = False):
    import __builtin__
    fout = __builtin__.open(fileName, 'w')
    d = {}
    d.update(header_dict)
    d['file type'] = 'ENVI Spectral Library'
    fout.write('ENVI\n')
    for k in d:
	_write_header_param(fout, k, d[k])
    fout.close()
    
def readEnviHdr(file):
    warn('readEnviHdr has been deprecated.  Use read_envi_header.',
	 DeprecationWarning)
    return read_envi_header(file)

def writeEnviHdr(fileName, header_dict, is_library = False):
    warn('writeEnviHdr has been deprecated.  Use write_envi_header.',
	 DeprecationWarning)
    return write_envi_header(fileName, header_dict, is_library)
