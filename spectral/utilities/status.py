#########################################################################
#
#   Status.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2001-2006 Thomas Boggs
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

class StatusDisplay:
    '''
    A class to sequentially display percentage completion of an iterative
    process on a single line.
    '''
    def __init__(self):
        self._pretext = ''
        self._overwrite = False
        self._percent_fmt = '% 5.1f'
        self._text_len = 0

    def display_percentage(self, text, percent=0.0, format='% 5.1f'):
        '''Called when initializing display of a process status.'''
        import sys
        from spectral import settings
        self._overwrite = True
        self._pretext = text
        self._percent_fmt = format
        text = self._pretext + self._percent_fmt % percent + '%'
        self._text_len = len(text)
        if settings.show_progress:
            sys.stdout.write(text)
            sys.stdout.flush()

    def update_percentage(self, percent):
        '''Called whenever an update of the displayed status is desired.'''
        import sys
        from spectral import settings
        if not settings.show_progress:
            return
        text = self._pretext + self._percent_fmt % percent + '%'
        sys.stdout.write('\b' * self._text_len)
        self._text_len = len(text)
        sys.stdout.write(text)
        sys.stdout.flush()

    def end_percentage(self, text='done'):
        '''Prints a final status and resumes normal text display.'''
        import sys
        from spectral import settings
        text = self._pretext + text
        sys.stdout.write('\b' * self._text_len)
        fmt = '%%-%ds\n' % self._text_len
        self._text_len = len(text)
        if settings.show_progress:
            sys.stdout.write(fmt % text)
            sys.stdout.flush()
        self._overwrite = False

    def write(self, text):
        '''
        Called to display text on a new line without interrupting
        progress display.
        '''
        import sys
        if self._overwrite and text != '\n':
            sys.stdout.write('\n')
        sys.stdout.write(text)
        sys.stdout.flush()
