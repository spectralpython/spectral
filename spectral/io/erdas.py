'''
Functions for reading Erdas files.
'''
# Following description accessed on 2011-01-25 at
#    http://www.pcigeomatics.com/cgi-bin/pcihlp/ERDASWR|IMAGE+FORMAT
#
# The ERDAS image file format contains a header record (128 bytes), followed by
# the image data. The image data is arranged in a Band Interleaved by Line
# (BIL) format. Each file is virtually unlimited in size - the file structure
# allows up to 274 billion bytes. The file consists of 512-byte records.
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


from __future__ import absolute_import, division, print_function, unicode_literals

import array
import numpy as np
import sys

import spectral as spy
from ..utilities.python23 import IS_PYTHON3, typecode
from .bilfile import BilFile
from .spyfile import find_file_path, InvalidFileError
from .spyfile import InvalidFileError

if IS_PYTHON3:
    import builtins
else:
    import __builtin__ as builtins



def open(file):
    '''
    Returns a SpyFile object for an ERDAS/Lan image file.

    Arguments:

        `file` (str):

            Name of the ERDAS/Lan image data file.

    Returns:

        A SpyFile object for the image file.

    Raises:

        spectral.io.spyfile.InvalidFileError
    '''

    # ERDAS 7.5 headers do not specify byte order so we'll guess little endian.
    # If any of the parameters look weird, we'll try again with big endian.

    class Params:
        pass
    p = Params()
    p.byte_order = 0

    file_path = find_file_path(file)

    lh = read_erdas_lan_header(find_file_path(file))
    if lh["nbands"] < 0 or lh["nbands"] > 512 or \
        lh["ncols"] < 0 or lh["ncols"] > 10000 or \
            lh["nrows"] < 0 or lh["nrows"] > 10000:
        p.byte_order = 1
        lh = read_erdas_lan_header(file_path, 1)

    p.filename = file_path
    p.nbands = lh["nbands"]
    p.ncols = lh["ncols"]
    p.nrows = lh["nrows"]
    p.offset = 128
    if lh["packing"] == 2:
        p.dtype = np.dtype('i2').str
    elif lh["packing"] == 0:
        p.dtype = np.dtype('i1').str
    elif lh["packing"] == 1:
        msg = '4-bit data type not supported in SPy ERDAS/Lan format handler.'
        raise InvalidFileError(msg)
    else:
        msg = 'Unexpected data type specified in ERDAS/Lan header.'
        raise InvalidFileError(msg)
    if spy.byte_order != 0:
        p.dtype = np.dtype(p.dtype).newbyteorder().str

    return BilFile(p, lh)


def read_erdas_lan_header(fileName, byte_order=0):
    '''Read parameters from a lan file header.

    Arguments:

        fileName (str):

            File to open.

        byte_order (int) [default 0]:

            Specifies whether to read as little (0) or big (1) endian.
    '''
    f = builtins.open(fileName, "rb")

    h = {}
    h["format"] = "lan"
    h["fileName"] = fileName
    h["sizeOfHeader"] = 128

    h["type"] = f.read(6)
    if h["type"] not in (b'HEAD74', b'HEADER'):
        raise InvalidFileError('Does not look like an ERDAS Lan header.')

    # Read all header data into arrays
    word = array.array(typecode('h'))
    dword = array.array(typecode('i'))
    float = array.array(typecode('f'))
    word.fromfile(f, 2)
    f.seek(16)
    if h["type"] == b'HEAD74':
        dword.fromfile(f, 4)
    else:
        float.fromfile(f, 4)
    f.seek(88)
    word.fromfile(f, 2)
    f.seek(106)
    word.fromfile(f, 1)
    float.fromfile(f, 5)

    if byte_order != spy.byte_order:
        word.byteswap()
        dword.byteswap()
        float.byteswap()

    # Unpack all header data
    h["packing"] = word.pop(0)
    h["nbands"] = word.pop(0)

    if h["type"] == b'HEAD74':
        h["ncols"] = dword.pop(0)
        h["nrows"] = dword.pop(0)
        h["pixel_xcoord"] = dword.pop(0)
        h["pixel_ycoord"] = dword.pop(0)
    else:
        h["ncols"] = int(float.pop(0))
        h["nrows"] = int(float.pop(0))
        h["pixel_xcoord"] = float.pop(0)
        h["pixel_ycoord"] = float.pop(0)

    h["map_type"] = word.pop(0)
    h["nclasses"] = word.pop(0)
    h["area_unit"] = word.pop(0)
    h["ypixel_size"] = float.pop()
    h["xpixel_size"] = float.pop()
    h["map_ycoord"] = float.pop()
    h["map_xcoord"] = float.pop()
    h["narea_units"] = float.pop()

    f.close()

    return h

