#########################################################################
#
#   python23.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2001-2014 Thomas Boggs
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
'''Functions for python 2/3 compatibility.'''

from __future__ import division, print_function, unicode_literals

import sys

IS_PYTHON3 = sys.version_info >= (3,)

def typecode(t):
    '''Typecode handling for array module.

    Python 3 expects a unicode character, whereas python 2 expects a byte char.

    Arguments:

        `t` (typecode string):

            An input for array.array.

    Return value:

        The input formatted for the appropriate python version.
    '''
    if IS_PYTHON3:
        return t
    else:
        return chr(ord(t))

if IS_PYTHON3:
    def is_string(s):
        return isinstance(s, (str, bytes))
else:
    def is_string(s):
        return isinstance(s, basestring)

# array.tostring is deprecated in python3
if IS_PYTHON3:
    tobytes = lambda array: array.tobytes()
    frombytes = lambda array, src: array.frombytes(src)
else:
    tobytes = lambda array: array.tostring()
    frombytes = lambda array, src: array.fromstring(src)
