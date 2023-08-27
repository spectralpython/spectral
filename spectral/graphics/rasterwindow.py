'''
Code for raster displays using wxPython.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import wx
from spectral.graphics.graphics import SpyWindow

from ..utilities.python23 import tobytes

logger = logging.getLogger('spectral')


class RasterWindow(wx.Frame, SpyWindow):
    '''
    RasterWindow is the primary wxWindows object for displaying SPy
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
        img.SetData(tobytes(rgb))
        self.bmp = img.ConvertToBitmap()
        self.kwargs = kwargs
        wx.Frame.__init__(self, parent, index, title,
                          wx.DefaultPosition)
        self.SetClientSizeWH(self.bmp.GetWidth(), self.bmp.GetHeight())
        wx.EVT_PAINT(self, self.on_paint)
        wx.EVT_LEFT_DCLICK(self, self.left_double_click)

    def on_paint(self, e):
        dc = wx.PaintDC(self)
        self.paint(dc)

    def paint(self, dc):

        dc.BeginDrawing()
        dc.DrawBitmap(self.bmp, 0, 0)
        # dc.Blit(0,0, bmp.GetWidth(), bmp.GetHeight(), mDC, 0, 0)
        dc.EndDrawing()

    def left_double_click(self, evt):
        from spectral import settings
        if "data source" in self.kwargs:
            logger.info('{}'.format((evt.GetY(), evt.GetX()))),
            settings.plotter.plot(self.kwargs["data source"],
                                  [evt.GetY(), evt.GetX()],
                                  source=self.kwargs["data source"])
