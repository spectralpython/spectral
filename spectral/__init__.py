#########################################################################
#
#   spectral/__init__.py - This file is part of the Spectral Python (SPy)
#   package.
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

from __future__ import division, print_function, unicode_literals

__version__ = '0.20'

import sys
if sys.byteorder == 'little':
    byte_order = 0   # little endian
else:
    byte_order = 1   # big endian

BSQ = 0
BIL = 1
BIP = 2

class SpyException(Exception):
    '''Base class for spectral module-specific exceptions.'''
    pass

from .spectral import (open_image, load_training_sets, save_training_sets,
                      settings, tile_image, spy_colors, BandInfo)
from .io import *
from .algorithms import *
from .graphics import *
from .database import *

# Import some submodules into top-level namespace
from .algorithms import detectors

from .spectral import _init
_init()
del _init
