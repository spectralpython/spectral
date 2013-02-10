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

__version__ = '0.9+'

# START_WX_APP is True and there is no current wx.App object when a GUI
# function is called, then an app object will be created.
START_WX_APP = True

import sys
if sys.byteorder == 'little':
    byte_order = 0   # little endian
else:
    byte_order = 1   # big endian

BSQ = 0
BIL = 1
BIP = 2

#from numpy import *
from spectral import image, load_training_sets, save_training_sets, settings, \
     tile_image, spy_colors, BandInfo
from io import *
from algorithms import *
from graphics import *
from database import *

try:
    import pylab
    from graphics import spypylab
    pylab.ion()
    spectral.settings.plotter = spypylab
    spectral.settings.viewer = graphics
except:
    warn('Unable to import orconfigure pylab plotter.  Spectrum plots will be '
	 'unavailable.', UserWarning)

import utilities.status
status = utilities.status.StatusDisplay()

