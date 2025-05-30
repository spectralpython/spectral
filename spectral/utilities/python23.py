'''
Functions for python 2/3 compatibility.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import sys

IS_PYTHON3 = sys.version_info >= (3,)


def typecode(t):
    '''Typecode handling for array module.

    Python 3 expects a unicode character, whereas python 2 expects a byte char.

    Arguments:

        `t` (typecode string):

            An input for array.array.

    Return value:

        The input formatted for the appropriate python version.
    '''
    if IS_PYTHON3:
        return t
    else:
        return chr(ord(t))


if IS_PYTHON3:
    def is_string(s):
        return isinstance(s, (str, bytes))
else:
    def is_string(s):
        return isinstance(s, basestring)  # noqa: F821


# array.tostring is deprecated in python3
if IS_PYTHON3:
    def tobytes(array): return array.tobytes()
    def frombytes(array, src): return array.frombytes(src)
else:
    def tobytes(array): return array.tostring()
    def frombytes(array, src): return array.fromstring(src)
