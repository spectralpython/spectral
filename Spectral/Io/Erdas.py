#########################################################################
#
#   Erdas.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2001 Thomas Boggs
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
Functions for reading Erdas files.
'''

def ErdasLan(file):
    '''
    Create a SpyFile object from an Erdas Lan file header.
    '''

    from BilFile import BilFile

    lh = ReadErdasLanHeader(file)

    class Params: pass
    p = Params()

    p.fileName = file
    p.nBands = lh["nBands"]
    p.nCols = lh["nCols"]
    p.nRows = lh["nRows"]
    p.offset = 128
    p.byteOrder = 1
    if lh["packing"] == 2:
	lh["typeCode"] = 'h'
        p.format = 'h'
        p.typecode = 's'
    else:
	lh["typeCode"] = 'b'
        p.format = 'b'
        p.typecode = '1'
    
    return BilFile(p, lh)    


def ReadErdasLanHeader(fileName):
    '''Read parameters from a lan file header.'''

    from exceptions import IOError
    from array import *
    word = array('h');  word.byteswap()
    dword = array('i'); dword.byteswap()
    float = array('f'); float.byteswap()

    f = open(fileName, "rb")

    h = {}
    h["format"] = "lan"
    h["fileName"] = fileName
    h["sizeOfHeader"] = 128

    h["type"] = f.read(6)
    if h["type"] != 'HEAD74':
        raise IOError, 'Does not look like an Erdas Lan header.'
    word.fromfile(f, 1)
    h["packing"] = word.pop()

    if h["packing"] == 2:
	h["typeCode"] = 'h'
    else:
	h["typeCode"] = 'c'

    word.fromfile(f, 1)
    h["nBands"] = word.pop()
    f.seek(16)
    dword.fromfile(f, 1)
    h["nCols"] = dword.pop()
    dword.fromfile(f, 1)
    h["nRows"] = dword.pop()
    dword.fromfile(f, 1)
    h["pixelXCoord"] = dword.pop()
    dword.fromfile(f, 1)
    h["pixelYCoord"] = dword.pop()
    f.seek(88)
    word.fromfile(f, 1)
    h["mapType"] = word.pop()
    word.fromfile(f, 1)
    h["nClasses"] = word.pop()
    f.seek(106)
    word.fromfile(f, 1)
    h["areaUnit"] = word.pop()
    float.fromfile(f, 5)
    h["yPixelSize"] = float.pop()
    h["xPixelSize"] = float.pop()
    h["mapYCoord"] = float.pop()
    h["mapXCoord"] = float.pop()
    h["nAreaUnits"] = float.pop()

    f.close()

    return h
