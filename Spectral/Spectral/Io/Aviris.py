#########################################################################
#
#   Aviris.py - This file is part of the Spectral Python (SPy) package.
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
Functions for handling AVIRIS image files.
'''

def Aviris(file):
    '''Creates a SpyFile object for an AVIRIS image file.'''

    from Spectral.Io.BipFile import BipFile
    import os
    from exceptions import IOError

    class Params: pass
    p = Params()

    p.fileName = file
    p.nBands = 224
    p.nCols = 614
    fileSize = os.stat(file)[6]
    if fileSize % 275072 != 0:
        raise IOError, 'File size not consitent with Aviris format.'
    p.nRows = fileSize / 275072
    p.format = 'h'
    p.typecode = 's'
    p.offset = 0
    p.byteOrder = 1
    metadata = {'default bands' : ['29', '18', '8']}

    return BipFile(p, metadata)    
