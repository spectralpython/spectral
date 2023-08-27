'''
Runs a set of unit tests for the spectral package.

To run all unit tests, type the following from the system command line:

    # python -m spectral.tests.run
'''
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from optparse import OptionParser

import spectral.tests


def parse_args():
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
        msg = '%d of %d tests FAILED.' % (spectral.tests._num_tests_failed,
                                          spectral.tests._num_tests_run)
    else:
        msg = 'All %d tests PASSED!' % spectral.tests._num_tests_run
    print('\n' + '-' * 72)
    print(msg)
    print('-' * 72)


if __name__ == '__main__':
    logging.getLogger('spectral').setLevel(logging.ERROR)
    parse_args()
    reset_stats()
    for test in spectral.tests.all_tests:
        test.run()
    print_summary()
