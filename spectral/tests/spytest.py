'''
Base class for all tests.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import sys

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
    import spectral.tests as tests

    def meth(self):
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
