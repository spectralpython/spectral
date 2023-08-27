'''
Code for package-level customization.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


class SpySettings:
    '''Run-time settings for the `spectral` module.

    After importing `spectral`, the settings object is referenced as
    `spectral.settings`.

    Noteworthy members:

        `WX_GL_DEPTH_SIZE` (integer, default 24):

            Sets the depth (in number of bits) for the OpenGL depth buffer.
            If calls to `view_cube` or `view_nd` result in windows with blank
            canvases, try reducing this value.

        `envi_support_nonlowercase_params` (bool, default False)

            By default, ENVI headers are read with parameter names converted
            to lower case. If this attribute is set to True, parameters will
            be read with original capitalization retained.

        `show_progress` (bool, default True):

            Indicates whether long-running algorithms should display progress
            to sys.stdout. It can be useful to set this value to False when
            SPy is embedded in another application (e.g., IPython Notebook).

        `imshow_figure_size` (2-tuple of integers, default `None`):

            Width and height (in inches) of windows opened with `imshow`. If
            this value is `None`, matplotlib's default size is used.

        `imshow_background_color` (3-tuple of integers, default (0,0,0)):

            Default color to use for masked pixels in `imshow` displays.

        `imshow_interpolation` (str, default `None`):

            Pixel interpolation to be used in imshow windows. If this value
            is `None`, matplotlib's default interpolation is used. Note that
            zoom windows always use "nearest" interpolation.

        `imshow_stretch`:

            Default RGB linear color stretch to perform.

        `imshow_stretch_all`:

            If True, each color channel limits are determined independently.

        `imshow_zoom_figure_width` (int, default `None`):

            Width of zoom windows opened from an imshow window. Since zoom
            windows are always square, this is also the window height. If this
            value is `None`, matplotlib's default window size is used.

        `imshow_zoom_pixel_width` (int, default 50):

            Number of source image pixel rows and columns to display in a
            zoom window.

        `imshow_float_cmap` (str, default "gray"):

            imshow color map to use with floating point arrays.

        `imshow_class_alpha` (float, default 0.5):

            alpha blending value to use for imshow class overlays

        `imshow_enable_rectangle_selector` (bool, default True):

            Whether to create the rectangle selection tool that enables
            interactive image pixel class labeling. On some OS/backend
            combinations, an exception may be raised when this object is
            created so disabling it allows imshow windows to be created without
            using the selector tool.

        `imshow_disable_mpl_callbacks` (bool, default True):

            If True, several matplotlib keypress event callbacks will be
            disabled to prevent conflicts with callbacks from SPy.  The
            matplotlib callbacks can be set back to their defaults by
            calling `matplotlib.rcdefaults()`.
    '''
    viewer = None
    plotter = None

    # If START_WX_APP is True and there is no current wx.App object when a
    # GUI function is called, then an app object will be created.
    START_WX_APP = True

    # Parameter used by GLCanvas objects in view_cube and view_nd. If the
    # canvas does not render, try reducing this value (e.g., 16).
    WX_GL_DEPTH_SIZE = 24

    envi_support_nonlowercase_params = False

    # Should algorithms show completion progress of algorithms?
    show_progress = True

    # imshow settings
    imshow_figure_size = None
    imshow_background_color = (0, 0, 0)
    imshow_interpolation = None
    imshow_stretch = (0.0, 1.0)
    imshow_stretch_all = True
    imshow_zoom_figure_width = None
    imshow_zoom_pixel_width = 50
    imshow_float_cmap = 'gray'
    imshow_class_alpha = 0.5
    imshow_enable_rectangle_selector = True
    imshow_disable_mpl_callbacks = True


# Default color table
spy_colors = np.array([[0, 0, 0],
                       [255, 0, 0],
                       [0, 255, 0],
                       [0, 0, 255],
                       [255, 255, 0],
                       [255, 0, 255],
                       [0, 255, 255],
                       [200, 100, 0],
                       [0, 200, 100],
                       [100, 0, 200],
                       [200, 0, 100],
                       [100, 200, 0],
                       [0, 100, 200],
                       [150, 75, 75],
                       [75, 150, 75],
                       [75, 75, 150],
                       [255, 100, 100],
                       [100, 255, 100],
                       [100, 100, 255],
                       [255, 150, 75],
                       [75, 255, 150],
                       [150, 75, 255],
                       [50, 50, 50],
                       [100, 100, 100],
                       [150, 150, 150],
                       [200, 200, 200],
                       [250, 250, 250],
                       [100, 0, 0],
                       [200, 0, 0],
                       [0, 100, 0],
                       [0, 200, 0],
                       [0, 0, 100],
                       [0, 0, 200],
                       [100, 100, 0],
                       [200, 200, 0],
                       [100, 0, 100],
                       [200, 0, 200],
                       [0, 100, 100],
                       [0, 200, 200]], np.int16)
