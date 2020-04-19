'''
A class for display completion status for long-running iterative operations.
'''

from __future__ import division, print_function, unicode_literals

import sys
from .. import settings

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
        self._overwrite = True
        self._pretext = text
        self._percent_fmt = format
        text = self._pretext + self._percent_fmt % percent + '%'
        self._text_len = len(text)
        if hasattr(sys, 'ps1') and settings.show_progress:
            sys.stdout.write(text)
            sys.stdout.flush()

    def update_percentage(self, percent):
        '''Called whenever an update of the displayed status is desired.'''
        if not (hasattr(sys, 'ps1') and settings.show_progress):
            return
        text = self._pretext + self._percent_fmt % percent + '%'
        sys.stdout.write('\b' * self._text_len)
        self._text_len = len(text)
        sys.stdout.write(text)
        sys.stdout.flush()

    def end_percentage(self, text='done'):
        '''Prints a final status and resumes normal text display.'''
        if not (hasattr(sys, 'ps1') and settings.show_progress):
            return
        text = self._pretext + text
        sys.stdout.write('\b' * self._text_len)
        fmt = '%%-%ds\n' % self._text_len
        self._text_len = len(text)
        sys.stdout.write(fmt % text)
        sys.stdout.flush()
        self._overwrite = False

    def write(self, text):
        '''
        Called to display text on a new line without interrupting
        progress display.
        '''
        if not (hasattr(sys, 'ps1') and settings.show_progress):
            return
        if self._overwrite and text != '\n':
            sys.stdout.write('\n')
        sys.stdout.write(text)
        sys.stdout.flush()
