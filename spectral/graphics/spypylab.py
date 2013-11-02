#########################################################################
#
#   spypylab.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2001-2013 Thomas Boggs
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
A module to use matplotlib for creating raster and spectral views.
'''

class ImageView(object):
    '''Class to manage events and data associated with `spectral.imshow`.'''

    def __init__(self, data, bands=None, **kwargs):

        import matplotlib.pyplot as plt
        from .graphics import get_rgb

        self.data = data
        self.bands = bands
        self.kwargs = kwargs

        (self.nrows, self.ncols) = self.data.shape[:2]

        self.spectrum_plot_fig_id = None
        self.parent = None
        self._on_parent_click_cid = None

    def show(self):
        '''Displays the image.'''
        import matplotlib.pyplot as plt
        from .graphics import get_rgb
        show_xaxis = True
        show_yaxis = True
        kwargs = self.kwargs.copy()
        
        if 'show_xaxis' in kwargs:
            show_xaxis = kwargs.pop('show_xaxis')
        if 'show_yaxis' in kwargs:
            show_yaxis = kwargs.pop('show_yaxis')
        rgb_kwargs = {}
        for k in ['stretch', 'stretch_all', 'bounds']:
            if k in kwargs:
                rgb_kwargs[k] = kwargs.pop(k)

        imshow_kwargs = {'cmap': 'gray', 'interpolation': 'none'}
        imshow_kwargs.update(kwargs)

        rgb = get_rgb(self.data, self.bands, **rgb_kwargs)

        # Allow matplotlib.imshow to apply a color scale to single-band image.
        if len(self.data.shape) == 2:
            rgb = rgb[:, :, 0]

        self.axesimage = plt.imshow(rgb, **imshow_kwargs)
        if show_xaxis == False:
            plt.gca().xaxis.set_visible(False)
        if show_yaxis == False:
            plt.gca().yaxis.set_visible(False)

        self._on_click_cid = self.axesimage.figure.canvas.mpl_connect(
            'button_press_event',
            self.on_click)

    def on_click(self, event):
        '''Callback for click event in the image display.'''
        if event.inaxes is not self.axesimage.axes:
            return
        (r, c) = (int(event.ydata + 0.5), int(event.xdata + 0.5))
        if r < 0 or r >= self.nrows or c < 0 or c >= self.ncols:
            return
        if event.button == 1:
            if event.key == 'control':
                if event.dblclick:
                    from spectral import settings
                    import matplotlib.pyplot as plt
                    if self.spectrum_plot_fig_id is None:
                        f = plt.figure()
                        self.spectrum_plot_fig_id = f.number
                    try:
                        f = plt.figure(self.spectrum_plot_fig_id)
                    except:
                        f = plt.figure()
                        self.spectrum_plot_fig_id = f.number
                    settings.plotter.plot(self.data[r, c], self.data)
                else:
                    self.print_pixel_info(r, c)
                
    def print_pixel_info(self, r, c):
        print '(row, col) = (%d, %d)' % (r, c)

    def set_parent(self, imageview):
        '''Makes the window dependent on mouse events in the parent window.'''
        self.unset_parent()
        self.parent = imageview
        self._on_parent_click_cid = \
            self.parent.axesimage.figure.canvas.mpl_connect(
                'button_press_event', self.on_parent_click)

    def unset_parent(self):
        if self.parent is not None:
            self.parent.axesimage.figure.canvas.mpl_disconnect(
                self._on_parent_click_cid)
            self.parent = None

    def on_parent_click(self, event):
        '''Callback for clicks in the image's parent window.'''
        try:
            if event.inaxes is not self.parent.axesimage.axes:
                return
            (r, c) = (int(event.ydata + 0.5), int(event.xdata + 0.5))
            if r < 0 or r >= self.nrows or c < 0 or c >= self.ncols:
                return
            if event.key == 'shift':
                self.pan_to(event.xdata, event.ydata)
        except:
            self.unset_parent()
        
    def pan_to(self, x, y):
        '''Centers view on pixel coordinate (x, y).'''
        (xmin, xmax) = self.axesimage.axes.get_xlim()
        (ymin, ymax) = self.axesimage.axes.get_ylim()
        xrange_2 = (xmax - xmin) / 2.0
        yrange_2 = (ymax - ymin) / 2.0
        self.axesimage.axes.set_xlim(x - xrange_2, x + xrange_2)
        self.axesimage.axes.set_ylim(y - yrange_2, y + yrange_2)
        self.axesimage.figure.canvas.draw()

    def open_zoom(self):
        import matplotlib.pyplot as plt
        kwargs = {'interpolation': 'none',
                  'extent': (-0.5, self.ncols - 0.5, self.nrows - 0.5, -0.5)}
        kwargs.update(self.kwargs)
        fig = plt.figure(figsize=(4,4))
        view = ImageView(self.data, self.bands, **kwargs)
        view.show()
        view.axesimage.axes.set_xlim(0, 50)
        view.axesimage.axes.set_ylim(50, 0)
        view.set_parent(self)
        return view

    def zoom(self, scale):
        '''Zooms view in/out (`scale` > 1 zooms in).'''
        (xmin, xmax) = self.axesimage.axes.get_xlim()
        (ymin, ymax) = self.axesimage.axes.get_ylim()
        x = (xmin + xmax) / 2.0
        y = (ymin + ymax) / 2.0
        dx = (xmax - xmin) / 2.0 / scale
        dy = (ymax - ymin) / 2.0 / scale

        self.axesimage.axes.set_xlim(x - dx, x + dx)
        self.axesimage.axes.set_ylim(y - dy, y + dy)
        self.axesimage.figure.canvas.draw()

def imshow(data, bands=None, **kwargs):
    '''A wrapper around matplotlib's imshow for multi-band images.

    Arguments:

        `data` (SpyFile or ndarray):

            Can have shape (R, C) or (R, C, B).

        `bands` (tuple of integers, optional)

            If `bands` has 3 values, the bands specified are extracted from
            `data` to be plotted as the red, green, and blue colors,
            respectively. If it contains a single value, then a single band
            will be extracted from the image.

    Keyword Arguments:

        `show_xaxis` (bool, default True):

            Indicates whether to display x-axis ticks and labels.

        `show_yaxis` (bool, default True):

            Indicates whether to display y-axis ticks and labels.

        `parent` (ImageView):

            If this keyword is given, events generated in `parent` can be
            used to affect the newly created ImageView. For example, if a
            shift+left-click event occurs in the parent window, the child
            window will be re-centered on the clicked pixel. This allows one
            to create a zoom window by using the matplotlib imshow zoom tool
            to zoom in on the child window, then shift+left-clicking in the
            parent window to adjust the zoomed location. `parent` should be
            an object returned by a previous call to `spectral.imshow`.

    This function is a wrapper around
    :func:`~spectral.graphics.graphics.get_rgb` and matplotlib's imshow.
    All keyword arguments other than those described above are passed on to
    the wrapped functions.

    This function defaults the color scale (imshow's "cmap" keyword) to
    "gray". To use imshow's default color scale, call this function with
    keyword `cmap=None`.
    '''
    if 'parent' in kwargs:
        parent = kwargs.pop('parent')
    else:
        parent = None
    view = ImageView(data, bands, **kwargs)
    if parent is not None:
        view.set_parent(parent)
    view.show()
    return view

def imshow_orig(data, bands=None, **kwargs):
    '''A wrapper around matplotlib's imshow for multi-band images.

    Arguments:

        `data` (SpyFile or ndarray):

            Can have shape (R, C) or (R, C, B).

        `bands` (tuple of integers, optional)

            If `bands` has 3 values, the bands specified are extracted from
            `data` to be plotted as the red, green, and blue colors,
            respectively. If it contains a single value, then a single band
            will be extracted from the image.

    Keyword Arguments:

        `show_xaxis` (bool, default True):

            Indicates whether to display x-axis ticks and labels.

        `show_yaxis` (bool, default True):

            Indicates whether to display y-axis ticks and labels.

        `parent` (ImageView):

            If this keyword is given, events generated in `parent` can be
            used to affect the newly created ImageView. For example, if a
            shift+left-click event occurs in the parent window, the child
            window will be re-centered on the clicked pixel. This allows one
            to create a zoom window by using the matplotlib imshow zoom tool
            to zoom in on the child window, then shift+left-clicking in the
            parent window to adjust the zoomed location. `parent` should be
            an object returned by a previous call to `spectral.imshow`.

    This function is a wrapper around
    :func:`~spectral.graphics.graphics.get_rgb` and matplotlib's imshow.
    All keyword arguments other than those described above are passed on to
    the wrapped functions.

    This function defaults the color scale (imshow's "cmap" keyword) to
    "gray". To use imshow's default color scale, call this function with
    keyword `cmap=None`.
    '''
    import matplotlib.pyplot as plt
    from .graphics import get_rgb

    show_xaxis = True
    show_yaxis = True
    if 'show_xaxis' in kwargs:
        show_xaxis = kwargs.pop('show_xaxis')
    if 'show_yaxis' in kwargs:
        show_yaxis = kwargs.pop('show_yaxis')
    if 'parent' in kwargs:
        parent = kwargs.pop('parent')
    else:
        parent = None

    rgb_kwargs = {}
    for k in ['stretch', 'stretch_all', 'bounds']:
        if k in kwargs:
            rgb_kwargs[k] = kwargs.pop(k)
    
    imshow_kwargs = {'cmap': 'gray', 'interpolation': 'none'}
    imshow_kwargs.update(kwargs)

    rgb = get_rgb(data, bands, **rgb_kwargs)

    # Allow matplotlib.imshow to apply a color scale to single-band image.
    if len(data.shape) == 2:
        rgb = rgb[:, :, 0]
    
    axesimage = plt.imshow(rgb, **imshow_kwargs)
    if show_xaxis == False:
        plt.gca().xaxis.set_visible(False)
    if show_yaxis == False:
        plt.gca().yaxis.set_visible(False)
    imageview = ImageView(axesimage, data)
    if parent is not None:
        imageview.set_parent(parent)
    return imageview

def plot(data, source=None):
    '''
    Creates an x-y plot.

    USAGE: plot(data)

    If data is a vector, all the values in data will be drawn in a
    single series. If data is a 2D array, each column of data will
    be drawn as a separate series.
    '''
    import pylab
    from numpy import shape
    import spectral

    s = shape(data)

    if source is not None and hasattr(source, 'bands'):
        xvals = source.bands.centers
    else:
        xvals = None

    if len(s) == 1:
        if not xvals:
            xvals = range(len(data))
        p = pylab.plot(xvals, data)
    elif len(s) == 2:
        if not xvals:
            xvals = range(s[1])
        p = pylab.plot(xvals, data[0, :])
        pylab.hold(1)
        for i in range(1, s[0]):
            p = pylab.plot(xvals, data[i, :])
    spectral._xyplot = p
    pylab.grid(1)
    if source is not None and hasattr(source, 'bands'):
        xlabel = source.bands.band_quantity
        if len(source.bands.band_unit) > 0:
            xlabel = xlabel + ' (' + source.bands.band_unit + ')'
        pylab.xlabel(xlabel)
    return p
