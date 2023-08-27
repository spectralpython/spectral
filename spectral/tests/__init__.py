'''
Package containing unit test modules for various functionality.

To run all unit tests, type the following from the system command line:

    # python -m spectral.tests.run
'''

# flake8: noqa

from __future__ import absolute_import, division, print_function, unicode_literals

# If abort_on_fail is True, an AssertionError will be raised when a unit test
# fails; otherwise, the failure will be printed to stdout and testing will
# continue.
abort_on_fail = True

# Summary stats of unit test execution
_num_tests_run = 0
_num_tests_failed = 0

# Subdirectory to be created for unit test files
testdir = 'spectral_test_files'

from . import database
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
from . import continuum

# List of all submodules to be run from the `run` submodule.
all_tests = [spyfile, memmap, iterators, transforms, envi, spymath, detectors,
             classifiers, dimensionality, spatial, database, continuum]
