#########################################################################
#
#   __init__.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2013 Thomas Boggs
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
'''Package containing unit test modules for various functionality.

To run all unit tests, type the following from the system command line:

    # python -m spectral.tests.run
'''

from __future__ import division, print_function, unicode_literals

# If abort_on_fail is True, an AssertionError will be raised when a unit test
# fails; otherwise, the failure will be printed to stdout and testing will
# continue.
abort_on_fail = True

# Summary stats of unit test execution
_num_tests_run = 0
_num_tests_failed = 0

# Subdirectory to be created for unit test files
testdir = 'spectral_test_files'

from . import spyfile
from . import transforms
from . import memmap
from . import envi
from . import spymath
from . import detectors
from . import classifiers
from . import dimensionality
from . import spatial
from . import iterators

# List of all submodules to be run from the `run` submodule.
all_tests = [spyfile, memmap, iterators, transforms, envi, spymath, detectors,
             classifiers, dimensionality, spatial]
