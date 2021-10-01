'''
Basic package setup and global imports.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

__version__ = '0.22.3'

import sys
if sys.byteorder == 'little':
    byte_order = 0   # little endian
else:
    byte_order = 1   # big endian

BSQ = 0
BIL = 1
BIP = 2

from .utilities.errors import SpyException
from .config import SpySettings, spy_colors
settings = SpySettings()

from .spectral import (open_image, load_training_sets, BandInfo)
from .io import *
from .algorithms import *
from .graphics import *
from .database import *

# Import some submodules into top-level namespace
from .algorithms import detectors

from .spectral import _init
_init()
del _init
