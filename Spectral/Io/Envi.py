#########################################################################
#
#   Envi.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2001-2006 Thomas Boggs
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
Code for creating SpyFile objects from ENVI headers.
'''

def ReadEnviHdr(file):
    '''
    USAGE: hdr = ReadEnviHeader(file)

    Reads a standard ENVI image file header and returns the parameters in
    a dictionary as strings.
    '''

    from string import find, split, strip
    from exceptions import IOError
    
    f = open(file, 'r')
    
    if find(f.readline(), "ENVI") == -1:
        f.close()
        raise IOError, "Not an ENVI header."

    lines = f.readlines()
    f.close()

    dict = {}
    i = 1
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


def EnviHdr(file, image = None):
    '''Creates a SpyFile object from an ENVI HDR file.'''

    import os
    from exceptions import IOError, TypeError
    from SpyFile import findFilePath

    headerPath = findFilePath(file)
    h = ReadEnviHdr(headerPath)
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
            exts = ['', '.img', '.IMG', '.dat', '.DAT']
            for ext in exts:
                testname = headerPathTitle + ext
                if os.path.isfile(testname):
                    image = testname
                    break
        if not image:
            raise IOError, 'Unable to determine image file name.'
    p.fileName = image

    #  Determine numeric data type
    if h["data type"] == '2':
        #  Int16
        p.format = 'h'
        p.typecode = 'h'
    elif h["data type"] == '1':
        #  char
        p.format = 'b'
        p.typecode = 'b'
    elif h["data type"] == '3':
        #  float32
        p.format = 'f'
        p.typecode = 'f'
    elif h["data type"] == '4':
        #  float32
        p.format = 'f'
        p.typecode = 'f'
    elif h["data type"] == '12':
        #  Int16
        p.format = 'H'
        p.typecode = 'H'
    else:
        #  Don't recognize this type code
        raise TypeError, 'Unrecognized data type code in header ' + \
              'file.  If you believe the header to be correct, please' + \
              'submit a bug report to have the type coded added.'

    #  Return the appropriate object type for the interleave format.
    inter = h["interleave"]
    if inter == 'bil' or inter == 'BIL':
        from Spectral.Io.BilFile import BilFile
        return BilFile(p, h)
    elif inter == 'bip' or inter == 'BIP':
        from Spectral.Io.BipFile import BipFile
        return BipFile(p, h)
    else:
        from Spectral.Io.BsqFile import BsqFile
        return BsqFile(p, h)

