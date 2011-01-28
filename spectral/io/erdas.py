#########################################################################
#
#   erdas.py - This file is part of the Spectral Python (SPy) package.
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
Functions for reading Erdas files.
'''
# Following description accessed on 2011-01-25 at
#    http://www.pcigeomatics.com/cgi-bin/pcihlp/ERDASWR|IMAGE+FORMAT
#
# The ERDAS image file format contains a header record (128 bytes), followed by
# the image data. The image data is arranged in a Band Interleaved by Line (BIL)
# format. Each file is virtually unlimited in size - the file structure allows
# up to 274 billion bytes. The file consists of 512-byte records.
# 
#                 ERDAS IMAGE FILE FORMAT
#  +----------------------------------------------------------+
#  |   Record 1 (bytes 1 to 128) Header                       |
#  |   --------------------------------                       |
#  |                                                          |
#  |    Bytes   Type   Contents                               |
#  |                                                          |
#  |     1- 6    ASCII  Descriptor         (HEAD74 or HEADER) |
#  |     7- 8    I*2    Type of data:0=8 bit /1=4 bit/2=16 bit|
#  |     9-10    I*2    Number of Channels                    |
#  |    11-16           Unused                                |
#  |    17-20    I*4    Number of Pixels, if HEAD74           |
#  |            (R*4    Number of Pixels, if HEADER)          |
#  |    21-24    I*4    Number of Lines,  if HEAD74           |
#  |            (R*4    Number of Lines,  if HEADER)          |
#  |    25-28    I*4    X-coordinate of 1st pixel, if HEAD74  |
#  |            (R*4    X-coordinate of 1st pixel, if HEADER) |
#  |    29-32    I*4    Y-coordinate of 1st pixel, if HEAD74  |
#  |            (R*4    Y-coordinate of 1st pixel, if HEADER) |
#  |    33-88           Unused                                |
#  |    89-90    I*2    Integer which indicates Map type      |
#  |    91-92    I*2    Number of classes in the data set     |
#  |    93-106          Unused                                |
#  |   107-108   I*2    Units of area of each pixel           |
#  |                    0=NONE, 1=ACRE, 2=HECTAR, 3=OTHER     |
#  |   109-112   R*4    Number of pixel area units            |
#  |   113-116   R*4    Map X-coordinate of upper left corner |
#  |   117-120   R*4    Map Y-coordinate of upper left corner |
#  |   121-124   R*4    X-pixel size                          |
#  |   125-128   R*4    Y-pixel size                          |
#  |                                                          |
#  |   Data files values begin in bytes 129 and cross over    |
#  |   record boundaries as necessary.                        |
#  |   Data are arranged in following order:                  |  
#  |                                                          |
#  |   L - Lines;  C - Channels;  P - Pixels per line;        |
#  |                                                          |
#  |   Pixels 1 through x of line 1, band 1                   |
#  |   Pixels 1 through x of line 1, band n                   |
#  |                                                          |
#  |   Pixels 1 through x of line 2, band 1                   |
#  |   Pixels 1 through x of line 2, band n                   |
#  |                                                          |
#  |   Pixels 1 through x of line y, band 1                   |
#  |   Pixels 1 through x of line y, band n                   |
#  +----------------------------------------------------------+


def open(file):
    '''
    Returns a SpyFile object for an ERDAS/Lan image file.
    
    Arguments:
    
        `file` (str):
	
	    Name of the ERDAS/Lan image data file.
	
    Returns:
    
	A SpyFile object for the image file.
	
    Raises:
    
	IOError
    '''

    from bilfile import BilFile
    from spyfile import findFilePath

    # ERDAS 7.5 headers do not specify byte order so we'll guess little endian.
    # If any of the parameters look weird, we'll try again with big endian.
    
    class Params: pass
    p = Params()
    p.byteOrder = 0

    lh = readErdasLanHeader(findFilePath(file))
    if lh["nBands"] < 0 or lh["nBands"] > 512 or \
       lh["nCols"] < 0 or lh["nCols"] > 10000 or \
       lh["nRows"] < 0 or lh["nRows"] > 10000:
	  p.byteOrder = 1
	  lh = readErdasLanHeader(findFilePath(file), 1)

    p.fileName = file
    p.nBands = lh["nBands"]
    p.nCols = lh["nCols"]
    p.nRows = lh["nRows"]
    p.offset = 128
    if lh["packing"] == 2:
	lh["typeCode"] = 'h'
        p.format = 'h'
        p.typecode = 'h'
    else:
	lh["typeCode"] = 'b'
        p.format = 'b'
        p.typecode = 'b'
    
    return BilFile(p, lh)    


def readErdasLanHeader(fileName, byteOrder=0):
    '''Read parameters from a lan file header.
    
    Arguments:
    
	fileName (str):
	
	    File to open.
	
	byteOrder (int) [default 0]:
	
	    Specifies whether to read as little (0) or big (1) endian.
    '''
    from exceptions import IOError
    from array import array
    import __builtin__
    import spectral

    f = __builtin__.open(fileName, "rb")

    h = {}
    h["format"] = "lan"
    h["fileName"] = fileName
    h["sizeOfHeader"] = 128

    h["type"] = f.read(6)
    if h["type"] not in ('HEAD74', 'HEADER'):
        raise IOError, 'Does not look like an ERDAS Lan header.'

    # Read all header data into arrays
    word = array('h')
    dword = array('i')
    float = array('f')
    word.fromfile(f, 2)
    f.seek(16)
    if h["type"] == 'HEAD74':
	dword.fromfile(f, 4)
    else:
	float.fromfile(f, 4)
    f.seek(88)
    word.fromfile(f, 2)
    f.seek(106)
    word.fromfile(f, 1)
    float.fromfile(f, 5)
    
    if byteOrder != spectral.byteOrder:
	word.byteswap()
	dword.byteswap()
	float.byteswap()

    # Unpack all header data
    h["packing"] = word.pop(0)
    if h["packing"] == 2:
	h["typeCode"] = 'h'
    elif h["packing"] == 1:
	raise Exception('4-bit data type not supported in SPy ERDAS/Lan format handler.')
    elif h["packing"] == 0:
	h["typeCode"] = 'b'
    else:
	raise Exception('Unexpected data type specified in ERDAS/Lan header.')
    h["nBands"] = word.pop(0)

    if h["type"] == 'HEAD74':
	h["nCols"] = dword.pop(0)
	h["nRows"] = dword.pop(0)
	h["pixelXCoord"] = dword.pop(0)
	h["pixelYCoord"] = dword.pop(0)
    else:
	h["nCols"] = int(float.pop(0))
	h["nRows"] = int(float.pop(0))
	h["pixelXCoord"] = float.pop(0)
	h["pixelYCoord"] = float.pop(0)

    h["mapType"] = word.pop(0)
    h["nClasses"] = word.pop(0)
    h["areaUnit"] = word.pop(0)
    h["yPixelSize"] = float.pop()
    h["xPixelSize"] = float.pop()
    h["mapYCoord"] = float.pop()
    h["mapXCoord"] = float.pop()
    h["nAreaUnits"] = float.pop()

    f.close()

    return h
