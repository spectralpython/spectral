#########################################################################
#
#   run.py - This file is part of the Spectral Python (SPy) package.
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
'''Runs a set of unit tests for the spectral package.

To run all unit tests, type the following from the system command line:

    # python -m spectral.tests.run
'''
from __future__ import division, print_function, unicode_literals

import spectral.tests

def parse_args():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-c', '--continue', dest='continue_tests',
                      action='store_true', default=False,
                      help='Continue with remaining tests after a '
                           'failed test.')
    (options, args) = parser.parse_args()
    spectral.tests.abort_on_fail = not options.continue_tests

def reset_stats():
    spectral.tests._num_tests_run = 0
    spectral.tests._num_tests_failed = 0

def print_summary():
    if spectral.tests._num_tests_failed > 0:
        msg =  '%d of %d tests FAILED.' % (spectral.tests._num_tests_failed,
                                           spectral.tests._num_tests_run)
    else:
        msg =  'All %d tests PASSED!' % spectral.tests._num_tests_run
    print('\n' + '-' * 72)
    print(msg)
    print('-' * 72)

if __name__ == '__main__':
    parse_args()
    reset_stats()
    for test in spectral.tests.all_tests:
        test.run()
    print_summary()
