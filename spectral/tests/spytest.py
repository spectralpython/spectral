#########################################################################
#
#   spytest.py - This file is part of the Spectral Python (SPy) package.
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


class SpyTest(object):
    '''Base class for test cases.'''
    def setup(self):
        pass

    def finish(self):
        pass


def test_method(method):
    '''Decorator function for unit tests.'''
    def meth(self):
        import spectral.tests as tests
        from spectral.tests import abort_on_fail
        print format('Testing ' + method.__name__.split('_', 1)[-1],
                     '.<40'),
        try:
            method(self)
            print 'OK'
            tests._num_tests_run += 1
        except AssertionError:
            print 'FAILED'
            tests._num_tests_failed += 1
            if tests.abort_on_fail:
                raise
    return meth
