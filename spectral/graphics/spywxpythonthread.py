#########################################################################
#
#   spywxpythonthread.py - This file is part of the Spectral Python (SPy)
#   package.
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
wxWindows code which executes in a separate thread from the main thread
of execution. This module handles disply of images and related events.
'''

from __future__ import division, print_function, unicode_literals

DEFAULT_X_SIZE = 600
DEFAULT_Y_SIZE = 600

from wx import *
#from Numeric import *
from spectral.graphics import *


#---------------------------------------------------------------------------
#wxEVT_VIEW_IMAGE = wxID_HIGHEST + 1
wxEVT_VIEW_IMAGE = 50002


def EVT_VIEW_IMAGE(win, func):
    win.Connect(-1, -1, wxEVT_VIEW_IMAGE, func)


class view_imageRequest(wx.PyEvent):
    '''A request for a new image.'''
    def __init__(self, rgb, **kwargs):
        wx.PyEvent.__init__(self)
        self.SetEventType(wxEVT_VIEW_IMAGE)
        self.rgb = rgb
        self.kwargs = kwargs


class HiddenCatcher(wx.Frame):
    '''
        The "catcher" frame in the second thread.
        It is invisible.  It's only job is to receive
        events from the main thread, and create
        the appropriate windows.
    '''
    def __init__(self):
        wx.Frame.__init__(self, None, -1, '')

        EVT_VIEW_IMAGE(self, self.view_image)
#        self.bmp = wxBitmap("/dos/myphotos/roll2/can.bmp",
#                            wxBITMAP_TYPE_BMP)

    def view_image(self, evt):
        if 'function' in evt.kwargs:
            frame = evt.kwargs['function']()
            frame.Show(True)
            self.app.SetTopWindow(frame)
            frame.Raise()
        else:
            frame = WxImageFrame(None, -1, evt.rgb, **evt.kwargs)
            frame.Show(True)
            self.app.SetTopWindow(frame)


class WxImageFrame(wx.Frame):
    '''
    WxImageFrame is the primary wxWindows object for displaying SPy
    images.  The frames also handle left double-click events by
    displaying an x-y plot of the spectrum for the associated pixel.
    '''
    def __init__(self, parent, index, rgb, **kwargs):
        if 'title' in kwargs:
            title = kwargs['title']
        else:
            title = 'SPy Image'
#        wxFrame.__init__(self, parent, index, "SPy Frame")
#        wxScrolledWindow.__init__(self, parent, index, style = wxSUNKEN_BORDER)

        img = wx.EmptyImage(rgb.shape[0], rgb.shape[1])
        img = wx.EmptyImage(rgb.shape[1], rgb.shape[0])
        img.SetData(rgb.tostring())
        self.bmp = img.ConvertToBitmap()
        self.kwargs = kwargs
        wx.Frame.__init__(self, parent, index, title,
                          wx.DefaultPosition)
        self.SetClientSizeWH(self.bmp.GetWidth(), self.bmp.GetHeight())
        EVT_PAINT(self, self.on_paint)
        EVT_LEFT_DCLICK(self, self.left_double_click)

    def on_paint(self, e):
        dc = wx.PaintDC(self)
        self.paint(dc)

    def paint(self, dc):

        # mDC = wxMemoryDC()
        # mDC.SelectObject(bmp)
        # mDC.DrawBitmap(bmp, 0, 0)

        dc.BeginDrawing()
        dc.DrawBitmap(self.bmp, 0, 0)
        # dc.Blit(0,0, bmp.GetWidth(), bmp.GetHeight(), mDC, 0, 0)
        dc.EndDrawing()

    def left_double_click(self, evt):
        print((evt.m_y, evt.m_x))
        from spectral import settings
        if "data source" in self.kwargs:
            settings.plotter.plot(self.kwargs["data source"][evt.m_y, evt.m_x],
                                  source=self.kwargs["data source"])


class WxImageServer(wx.App):
    '''
    An image server built on wxPython.  This image server runs in a
    separate thread, displaying raster images and handling events
    related to the images.  DO NOT construct a WxImageServer object
    directly. Call StartWxImageServer instead.
    '''

    def OnInit(self):
        catcher = HiddenCatcher()
        catcher.app = self
        #self.SetTopWindow(catcher)
        self.catcher = catcher
        return True
