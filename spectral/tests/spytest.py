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


from __future__ import division, print_function, unicode_literals

import collections

class SpyTest(object):
    '''Base class for test cases.

    Test classes are created by sub-classing SpyTest and defining methods
    whose names start with "test_".
    '''
    def setup(self):
        '''Method to be run before derived class test methods are called.'''
        pass

    def finish(self):
        '''Method run after all test methods have run.'''
        pass

    def run(self):
        '''Runs all "test_*" methods in a derived class.

        Before running subclass test_ methods, the `startup` method will be
        called. After all test_ methods have been run, the `finish` method
        is called.
        '''
        import spectral.tests as tests
        from spectral.tests import abort_on_fail
        import sys
        self.setup()
        class NullStdOut(object):
            def write(*args, **kwargs):
                pass
            def flush(self):
                pass
        null = NullStdOut()
        methods = [getattr(self, s) for s in sorted(dir(self)) if s.startswith('test_')]
        methods = [m for m in methods if isinstance(m, collections.Callable)]
        stdout = sys.stdout
        for method in methods:
            print(format('Testing ' + method.__name__.split('_', 1)[-1],
                         '.<60'), end=' ')
            tests._num_tests_run += 1
            try:
                sys.stdout = null
                method()
                stdout.write('OK\n')
            except AssertionError:
                stdout.write('FAILED\n')
                tests._num_tests_failed += 1
                if tests.abort_on_fail:
                    raise
            finally:
                sys.stdout = stdout
        self.finish()

# The following test method is now deprecated and should no longer be used.

def test_method(method):
    '''Decorator function for unit tests.'''
    def meth(self):
        import spectral.tests as tests
        from spectral.tests import abort_on_fail
        print(format('Testing ' + method.__name__.split('_', 1)[-1],
                     '.<40'), end=' ')
        try:
            method(self)
            print('OK')
            tests._num_tests_run += 1
        except AssertionError:
            print('FAILED')
            tests._num_tests_failed += 1
            if tests.abort_on_fail:
                raise
    return meth
