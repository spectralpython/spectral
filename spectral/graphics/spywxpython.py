#########################################################################
#
#   spywxpython.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2001-2010 Thomas Boggs
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

'''
Classes and functions for viewing/manipulating images using wxWindows.
In order to use wxWindows and still have a command line interface,
wxWindows must be imported in a separate thread and all GUI objects
must be referenced in that thread.  Thus, much of the actual GUI code
is in SpyWxPythonThread.py.
'''

from __future__ import division, print_function, unicode_literals

viewer = None


class SpyWxPythonThreadStarter:
    def start(self):
        '''Starts the GUI thread.'''
        import _thread
        import time
        _thread.start_new_thread(self.run, ())

    def run(self):
        '''
        This is the first function executed in the wxWindows thread.
        It creates the wxApp and starts the main event loop.
        '''
        from .spywxpythonthread import WxImageServer
        self.app = WxImageServer(0)
        self.app.MainLoop()

    def view(self, rgb, **kwargs):
        '''Sends a view request to the wxWindows thread.'''

        from . import spywxpythonthread
        evt = spywxpythonthread.view_imageRequest(rgb, **kwargs)
        spywxpythonthread.wx.PostEvent(self.app.catcher, evt)


def init():
    global viewer
    viewer = SpyWxPythonThreadStarter()
    viewer.start()


def view(*args, **kwargs):
    '''Displays an image in a wxWindows frame.'''

    from . import graphics
    from spectral.spectral import Image
    import numpy as np

    rgb = graphics.get_rgb(*args, **kwargs)

    # To plot pixel spectrum on double-click, create a reference
    # back to the original SpyFile object.
    if isinstance(args[0], Image):
        kwargs["data source"] = args[0]

    if "colors" not in kwargs:
        rgb = (rgb * 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)
    viewer.view(rgb, **kwargs)
