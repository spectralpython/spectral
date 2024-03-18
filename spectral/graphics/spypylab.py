'''
Code to use matplotlib for creating raster and spectral views.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = ['ImageView', 'imshow']

import numpy as np
import warnings

_mpl_callbacks_checked = False


def check_disable_mpl_callbacks():
    '''Disables matplotlib key event handlers, if appropriate.'''
    import matplotlib as mpl
    from spectral import settings
    global _mpl_callbacks_checked
    if _mpl_callbacks_checked is True or \
      settings.imshow_disable_mpl_callbacks is False:
        return
    _mpl_callbacks_checked = True
    mpl.rcParams['keymap.back'] = ''
    mpl.rcParams['keymap.xscale'] = ''
    mpl.rcParams['keymap.yscale'] = ''
    mpl.rcParams['keymap.home'] = 'r'


def xy_to_rowcol(x, y):
    '''Converts image (x, y) coordinate to pixel (row, col).'''
    return (int(y + 0.5), int(x + 0.5))


def rowcol_to_xy(r, c):
    '''Converts pixel (row, col) coordinate to (x, y) of pixel center.'''
    return (float(c), float(r))


class MplCallback(object):
    '''Base class for callbacks using matplotlib's CallbackRegistry.

    Behavior of MplCallback objects can be customized by providing a callable
    object to the constructor (or `connect` method) or by defining a
    `handle_event` method in a subclass.
    '''
    # If the following class attribute is False, callbacks will silently
    # disconnect when an exception is encountered during event processing
    # (e.g., if an associated window has been closed) . If it is True, the
    # associated exception will be rethrown.
    raise_event_exceptions = False
    show_events = False

    def __init__(self, registry=None, event=None, callback=None):
        '''
         Arguments:

            registry (ImageView, CallbackRegistry, or FigureCanvas):

                The object that will generate the callback. If the argument is
                an ImageView, the callback will be bound to the associated
                FigureCanvas.

            event (str):

                The event type for which callbacks should be generated.

            callback (callable):

                An optional callable object to handle the event. If not
                provided, the `handle_event` method of the MplCallback will
                be called to handle the event (this method must be defined by
                a derived class if `callback` is not provided.

        Note that these arguments can be deferred until `MplCallback.connect`
        is called.
        '''
        self.set_registry(registry)
        self.event = event
        self.callback = callback
        self.cid = None
        self.is_connected = False
        self.children = []

    def set_registry(self, registry=None):
        '''
        Arguments:

            registry (ImageView, CallbackRegistry, or FigureCanvas):

                The object that will generate the callback. If the argument is
                an ImageView, the callback will be bound to the associated
                FigureCanvas.
        '''
        from matplotlib.cbook import CallbackRegistry
        if isinstance(registry, CallbackRegistry):
            self.registry = registry
        elif isinstance(registry, ImageView):
            self.registry = registry.axes.figure.canvas
        else:
            self.registry = registry

    def connect(self, registry=None, event=None, callback=None):
        '''Binds the callback to the registry and begins receiving events.

         Arguments:

            registry (ImageView, CallbackRegistry, or FigureCanvas):

                The object that will generate the callback. If the argument is
                an ImageView, the callback will be bound to the associated
                FigureCanvas.

            event (str):

                The event type for which callbacks should be generated.

            callback (callable):

                An optional callable object to handle the event. If not
                provided, the `handle_event` method of the MplCallback will
                be called to handle the event (this method must be defined by
                a derived class if `callback` is not provided.

        Note that these arguments can also be provided to the constructor.
        '''
        from matplotlib.cbook import CallbackRegistry
        if self.is_connected:
            raise Exception('Callback is already connected.')
        if registry is not None:
            self.set_registry(registry)
        if event is not None:
            self.event = event
        if callback is not None:
            self.callback = callback
        if isinstance(self.registry, CallbackRegistry):
            self.cid = self.registry.connect(self.event, self)
        elif isinstance(self.registry, ImageView):
            self.cid = self.registry.connect(self.event, self)
        else:
            # Assume registry is an MPL canvas
            self.cid = self.registry.mpl_connect(self.event, self)
        self.is_connected = True
        for c in self.children:
            c.connect()

    def disconnect(self):
        '''Stops the callback from receiving events.'''
        from matplotlib.cbook import CallbackRegistry
        if isinstance(self.registry, CallbackRegistry):
            self.registry.disconnect(self.cid)
        else:
            # Assume registry is an MPL canvas
            self.registry.mpl_disconnect(self.cid)
        self.is_connected = False
        self.cid = None
        for c in self.children:
            c.disconnect()

    def __call__(self, *args, **kwargs):
        if self.callback is not None:
            try:
                self.callback(*args, **kwargs)
            except Exception as e:
                self.disconnect()
                if self.raise_event_exceptions:
                    raise e
        else:
            try:
                self.handle_event(*args, **kwargs)
            except Exception as e:
                self.disconnect()
                if self.raise_event_exceptions:
                    raise e


class ImageViewCallback(MplCallback):
    '''Base class for callbacks that operate on ImageView objects.'''
    def __init__(self, view, *args, **kwargs):
        super(ImageViewCallback, self).__init__(*args, **kwargs)
        self.view = view


class ParentViewPanCallback(ImageViewCallback):
    '''A callback to pan an image based on a click in another image.'''
    def __init__(self, child, parent, *args, **kwargs):
        '''
        Arguments:

            `child` (ImageView):

                The view that will be panned based on a parent click event.

            `parent` (ImageView):

                The view whose click location will cause the child to pan.

        See ImageViewCallback and MplCallback for additional arguments.
        '''
        super(ParentViewPanCallback, self).__init__(parent, *args, **kwargs)
        self.child = child

    def handle_event(self, event):
        if self.show_events:
            print(event, 'key = %s' % event.key)
        if event.inaxes is not self.view.axes:
            return
        (r, c) = xy_to_rowcol(event.xdata, event.ydata)
        (nrows, ncols) = self.view._image_shape
        if r < 0 or r >= nrows or c < 0 or c >= ncols:
            return
        kp = KeyParser(event.key)
        if event.button == 1 and kp.mods_are('ctrl'):
            self.child.pan_to(event.ydata, event.xdata)

    def connect(self):
        super(ParentViewPanCallback, self).connect(registry=self.view,
                                                   event='button_press_event')


class ImageViewKeyboardHandler(ImageViewCallback):
    '''Default handler for keyboard events in an ImageView.'''
    def __init__(self, view, *args, **kwargs):
        super(ImageViewKeyboardHandler, self).__init__(view,
                                                       registry=view,
                                                       event='key_press_event',
                                                       *args, **kwargs)
        self.cb_key_release = ImageViewCallback(view, registry=view,
                                                event='key_release_event',
                                                callback=self.on_key_release,
                                                *args, **kwargs)
        # Must add to children member to automatically connect/disconnect.
        self.children.append(self.cb_key_release)
        self.idstr = ''

    def on_key_release(self, event):
        if self.show_events:
            print('key = %s' % event.key)
        kp = KeyParser(event.key)
        key = kp.key
        if key is None and self.view.selector is not None and \
          self.view.selector.get_active() and kp.mods_are('shift'):
            print('Resetting selection.')
            self.view.selector.eventpress = None
            self.view.selector.set_active(False)
            self.view.selection = None
            self.view.selector.to_draw.set_visible(False)
            self.view.refresh()

    def handle_event(self, event):
        from spectral import settings
        if self.show_events:
            print('key = %s' % event.key)
        kp = KeyParser(event.key)
        key = kp.key

        #-----------------------------------------------------------
        # Handling for keyboard input related to class ID assignment
        #-----------------------------------------------------------

        if key is None and kp.mods_are('shift') and \
          self.view.selector is not None:
            # Rectangle selector is active while shift key is pressed
            self.view.selector.set_active(True)
            return

        if key in [str(i) for i in range(10)] and self.view.selector is not None:
            if self.view.selection is None:
                print('Select an image region before assigning a class ID.')
                return
            if len(self.idstr) > 0 and self.idstr[-1] == '!':
                print('Cancelled class ID assignment.')
                self.idstr = ''
                return
            else:
                self.idstr += key
                return

        if key == 'enter' and self.view.selector is not None:
            if self.view.selection is None:
                print('Select an image region before assigning a class ID.')
                return
            if len(self.idstr) == 0:
                print('Enter a numeric class ID before assigning a class ID.')
                return
            if self.idstr[-1] != '!':
                print('Press ENTER again to assign class %s to pixel ' \
                  'region [%d:%d, %d:%d]:' \
                  % ((self.idstr,) + tuple(self.view.selection)))
                self.idstr += '!'
                return
            else:
                i = int(self.idstr[:-1])
                n = self.view.label_region(self.view.selection, i)
                if n == 0:
                    print('No pixels reassigned.')
                else:
                    print('%d pixels reassigned to class %d.' % (n, i))
                self.idstr = ''
                return

        if len(self.idstr) > 0:
            self.idstr = ''
            print('Cancelled class ID assignment.')

        #-----------------------------------------------------------
        # General keybinds
        #-----------------------------------------------------------

        if key == 'a' and self.view.display_mode == 'overlay':
            self.view.class_alpha = max(self.view.class_alpha - 0.05, 0)
        elif key == 'A' and self.view.display_mode == 'overlay':
            self.view.class_alpha = min(self.view.class_alpha + 0.05, 1)
        elif key == 'c':
            if self.view.classes is not None:
                self.view.set_display_mode('classes')
        elif key == 'C':
            if self.view.classes is not None \
              and self.view.data_axes is not None:
                self.view.set_display_mode('overlay')
        elif key == 'd':
            if self.view.data_axes is not None:
                self.view.set_display_mode('data')
        elif key == 'h':
            self.print_help()
        elif key == 'i':
            if self.view.interpolation == 'nearest':
                self.view.interpolation = settings.imshow_interpolation
            else:
                self.view.interpolation = 'nearest'
        elif key == 'z':
            self.view.open_zoom()

    def print_help(self):
        print()
        print('Mouse Functions:')
        print('----------------')
        print('ctrl+left-click          ->   pan zoom window to pixel')
        print('shift+left-click&drag    ->   select rectangular image region')
        print('left-dblclick            ->   plot pixel spectrum')

        print()
        print('Keybinds:')
        print('---------')
        print('0-9     -> enter class ID for image pixel labeling')
        print('ENTER   -> apply specified class ID to selected rectangular region')
        print('a/A     -> decrease/increase class overlay alpha value')
        print('c       -> set display mode to "classes" (if classes set)')
        print('C       -> set display mode to "overlay" (if data and ' \
                          'classes set)')
        print('d       -> set display mode to "data" (if data set)')
        print('h       -> print help message')
        print('i       -> toggle pixel interpolation between "nearest" and ' \
                          'SPy default.')
        print('z       -> open zoom window')
        print()
        print('See matplotlib imshow documentation for addition key binds.')
        print()


class KeyParser(object):
    '''Class to handle ambiguities in matplotlib event key values.'''
    aliases = {'ctrl': ['ctrl', 'control'],
               'alt': ['alt'],
               'shift': ['shift'],
               'super': ['super']}

    def __init__(self, key_str=None):
        self.reset()
        if key_str is not None:
            self.parse(key_str)

    def reset(self):
        self.key = None
        self.modifiers = set()

    def parse(self, key_str):
        '''Extracts the key value and modifiers from a string.'''
        self.reset()
        if key_str is None:
            return
        tokens = key_str.split('+')
        for token in tokens[:-1]:
            mods = self.get_token_modifiers(token)
            if len(mods) == 0:
                raise ValueError('Unrecognized modifier: %s' % repr(token))
            self.modifiers.update(mods)
        # For the final token, need to determine if it is a key or modifier
        mods = self.get_token_modifiers(tokens[-1])
        if len(mods) > 0:
            self.modifiers.update(mods)
        else:
            self.key = tokens[-1]

    def has_mod(self, m):
        '''Returns True if `m` is one of the modifiers.'''
        return m in self.modifiers

    def mods_are(self, *args):
        '''Return True if modifiers are exactly the ones specified.'''
        for a in args:
            if a not in self.modifiers:
                return False
        return True

    def get_token_modifiers(self, token):
        mods = set()
        for (modifier, aliases) in list(self.aliases.items()):
            if token in aliases:
                mods.add(modifier)
        return mods


class ImageViewMouseHandler(ImageViewCallback):
    def __init__(self, view, *args, **kwargs):
        super(ImageViewMouseHandler, self).__init__(view,
                                                    registry=view,
                                                    event='button_press_event',
                                                    *args, **kwargs)

    def handle_event(self, event):
        '''Callback for click event in the image display.'''
        if self.show_events:
            print(event, ', key = %s' % event.key)
        if event.inaxes is not self.view.axes:
            return
        (r, c) = (int(event.ydata + 0.5), int(event.xdata + 0.5))
        (nrows, ncols) = self.view._image_shape
        if r < 0 or r >= nrows or c < 0 or c >= ncols:
            return
        kp = KeyParser(event.key)
        if event.button == 1:
            if event.dblclick and kp.key is None:
                if self.view.source is not None:
                    from spectral import settings
                    import matplotlib.pyplot as plt
                    if self.view.spectrum_plot_fig_id is None:
                        f = plt.figure()
                        self.view.spectrum_plot_fig_id = f.number
                    try:
                        f = plt.figure(self.view.spectrum_plot_fig_id)
                    except:
                        f = plt.figure()
                        self.view.spectrum_plot_fig_id = f.number
                    s = f.gca()
                    settings.plotter.plot(self.view.source[r, c],
                                          self.view.source)
                    s.xaxis.axes.relim()
                    s.xaxis.axes.autoscale(True)
                    f.canvas.draw()


class SpyMplEvent(object):
    def __init__(self, name):
        self.name = name


class ImageView(object):
    '''Class to manage events and data associated with image raster views.

    In most cases, it is more convenient to simply call :func:`~spectral.graphics.spypylab.imshow`,
    which creates, displays, and returns an :class:`ImageView` object. Creating
    an :class:`ImageView` object directly (or creating an instance of a subclass)
    enables additional customization of the image display (e.g., overriding
    default event handlers). If the object is created directly, call the
    :meth:`show` method to display the image. The underlying image display
    functionality is implemented via :func:`matplotlib.pyplot.imshow`.
    '''

    selector_rectprops = dict(facecolor='red', edgecolor='black',
                              alpha=0.5, fill=True)
    selector_lineprops = dict(color='black', linestyle='-',
                              linewidth=2, alpha=0.5)

    def __init__(self, data=None, bands=None, classes=None, source=None,
                 **kwargs):
        '''
        Arguments:

            `data` (ndarray or :class:`SpyFile`):

                The source of RGB bands to be displayed. with shape (R, C, B).
                If the shape is (R, C, 3), the last dimension is assumed to
                provide the red, green, and blue bands (unless the `bands`
                argument is provided). If :math:`B > 3` and `bands` is not
                provided, the first, middle, and last band will be used.

            `bands` (triplet of integers):

                Specifies which bands in `data` should be displayed as red,
                green, and blue, respectively.

            `classes` (ndarray of integers):

                An array of integer-valued class labels with shape (R, C). If
                the `data` argument is provided, the shape must match the first
                two dimensions of `data`.

            `source` (ndarray or :class:`SpyFile`):

                The source of spectral data associated with the image display.
                This optional argument is used to access spectral data (e.g., to
                generate a spectrum plot when a user double-clicks on the image
                display.

        Keyword arguments:

            Any keyword that can be provided to :func:`~spectral.graphics.graphics.get_rgb`
            or :func:`matplotlib.imshow`.
        '''

        import spectral
        from spectral import settings
        self.is_shown = False
        self.imshow_data_kwargs = {'cmap': settings.imshow_float_cmap}
        self.rgb_kwargs = {}
        self.imshow_class_kwargs = {'zorder': 1}

        self.data = data
        self.data_rgb = None
        self.data_rgb_meta = {}
        self.classes = None
        self.class_rgb = None
        self.source = None
        self.bands = bands
        self.data_axes = None
        self.class_axes = None
        self.axes = None
        self._image_shape = None
        self.display_mode = None
        self._interpolation = None
        self.selection = None
        self.interpolation = kwargs.get('interpolation',
                                        settings.imshow_interpolation)

        if data is not None:
            self.set_data(data, bands, **kwargs)
        if classes is not None:
            self.set_classes(classes, **kwargs)
        if source is not None:
            self.set_source(source)

        self.class_colors = spectral.spy_colors

        self.spectrum_plot_fig_id = None
        self.parent = None
        self.selector = None
        self._on_parent_click_cid = None
        self._class_alpha = settings.imshow_class_alpha

        # Callbacks for events associated specifically with this window.
        self.callbacks = None

        # A sharable callback registry for related windows. If this
        # CallbackRegistry is set prior to calling ImageView.show (e.g., by
        # setting it equal to the `callbacks_common` member of another
        # ImageView object), then the registry will be shared. Otherwise, a new
        # callback registry will be created for this ImageView.
        self.callbacks_common = None

        check_disable_mpl_callbacks()

    def set_data(self, data, bands=None, **kwargs):
        '''Sets the data to be shown in the RGB channels.

        Arguments:

            `data` (ndarray or SpyImage):

                If `data` has more than 3 bands, the `bands` argument can be
                used to specify which 3 bands to display. `data` will be
                passed to `get_rgb` prior to display.

            `bands` (3-tuple of int):

                Indices of the 3 bands to display from `data`.

        Keyword Arguments:

            Any valid keyword for `get_rgb` or `matplotlib.imshow` can be
            given.
        '''
        from .graphics import _get_rgb_kwargs

        self.data = data
        self.bands = bands

        rgb_kwargs = {}
        for k in _get_rgb_kwargs:
            if k in kwargs:
                rgb_kwargs[k] = kwargs.pop(k)
        self.set_rgb_options(**rgb_kwargs)

        self._update_data_rgb()

        if self._image_shape is None:
            self._image_shape = data.shape[:2]
        elif data.shape[:2] != self._image_shape:
            raise ValueError('Image shape is inconsistent with previously ' \
                             'set data.')
        self.imshow_data_kwargs.update(kwargs)
        if 'interpolation' in self.imshow_data_kwargs:
            self.interpolation = self.imshow_data_kwargs['interpolation']
            self.imshow_data_kwargs.pop('interpolation')

        if len(kwargs) > 0 and self.is_shown:
            msg = 'Keyword args to set_data only have an effect if ' \
              'given before the image is shown.'
            warnings.warn(UserWarning(msg))
        if self.is_shown:
            self.refresh()

    def set_rgb_options(self, **kwargs):
        '''Sets parameters affecting RGB display of data.

        Accepts any keyword supported by :func:`~spectral.graphics.graphics.get_rgb`.
        '''
        from .graphics import _get_rgb_kwargs

        for k in kwargs:
            if k not in _get_rgb_kwargs:
                raise ValueError('Unexpected keyword: {0}'.format(k))
        self.rgb_kwargs = kwargs.copy()
        if self.is_shown:
            self._update_data_rgb()
            self.refresh()

    def _update_data_rgb(self):
        '''Regenerates the RGB values for display.'''
        from .graphics import get_rgb_meta

        (self.data_rgb, self.data_rgb_meta) = \
          get_rgb_meta(self.data, self.bands, **self.rgb_kwargs)

        # If it is a gray-scale image, only keep the first RGB component so
        # matplotlib imshow's cmap can still be used.
        if self.data_rgb_meta['mode'] == 'monochrome' and \
           self.data_rgb.ndim == 3:
            self.data_rgb = self.data_rgb[:, :, 0]

    def set_classes(self, classes, colors=None, **kwargs):
        '''Sets the array of class values associated with the image data.

        Arguments:

            `classes` (ndarray of int):

                `classes` must be an integer-valued array with the same
                number rows and columns as the display data (if set).

            `colors`: (array or 3-tuples):

                Color triplets (with values in the range [0, 255]) that
                define the colors to be associated with the integer indices
                in `classes`.

        Keyword Arguments:

            Any valid keyword for `matplotlib.imshow` can be provided.
        '''
        from .graphics import _get_rgb_kwargs
        self.classes = classes
        if classes is None:
            return
        if self._image_shape is None:
            self._image_shape = classes.shape[:2]
        elif classes.shape[:2] != self._image_shape:
            raise ValueError('Class data shape is inconsistent with ' \
                             'previously set data.')
        if colors is not None:
            self.class_colors = colors

        kwargs = dict([item for item in list(kwargs.items()) if item[0] not in \
                       _get_rgb_kwargs])
        self.imshow_class_kwargs.update(kwargs)

        if 'interpolation' in self.imshow_class_kwargs:
            self.interpolation = self.imshow_class_kwargs['interpolation']
            self.imshow_class_kwargs.pop('interpolation')

        if len(kwargs) > 0 and self.is_shown:
            msg = 'Keyword args to set_classes only have an effect if ' \
              'given before the image is shown.'
            warnings.warn(UserWarning(msg))
        if self.is_shown:
            self.refresh()

    def set_source(self, source):
        '''Sets the image data source (used for accessing spectral data).

        Arguments:

            `source` (ndarray or :class:`SpyFile`):

                The source for spectral data associated with the view.
        '''
        self.source = source

    def show(self, mode=None, fignum=None):
        '''Renders the image data.

        Arguments:

            `mode` (str):

                Must be one of:

                    "data":          Show the data RGB

                    "classes":       Shows indexed color for `classes`

                    "overlay":       Shows class colors overlaid on data RGB.

                If `mode` is not provided, a mode will be automatically
                selected, based on the data set in the ImageView.

            `fignum` (int):

                Figure number of the matplotlib figure in which to display
                the ImageView. If not provided, a new figure will be created.
        '''
        import matplotlib.pyplot as plt
        from spectral import settings

        if self.is_shown:
            msg = 'ImageView.show should only be called once.'
            warnings.warn(UserWarning(msg))
            return

        set_mpl_interactive()

        kwargs = {}
        if fignum is not None:
            kwargs['num'] = fignum
        if settings.imshow_figure_size is not None:
            kwargs['figsize'] = settings.imshow_figure_size
        plt.figure(**kwargs)

        if self.data_rgb is not None:
            self.show_data()
        if self.classes is not None:
            self.show_classes()

        if mode is None:
            self._guess_mode()
        else:
            self.set_display_mode(mode)

        self.axes.format_coord = self.format_coord

        self.init_callbacks()
        self.is_shown = True

    def init_callbacks(self):
        '''Creates the object's callback registry and default callbacks.'''
        from spectral import settings
        from matplotlib.cbook import CallbackRegistry

        self.callbacks = CallbackRegistry()

        # callbacks_common may have been set to a shared external registry
        # (e.g., to the callbacks_common member of another ImageView object). So
        # don't create it if it has already been set.
        if self.callbacks_common is None:
            self.callbacks_common = CallbackRegistry()

        # Keyboard callback
        self.cb_mouse = ImageViewMouseHandler(self)
        self.cb_mouse.connect()

        # Mouse callback
        self.cb_keyboard = ImageViewKeyboardHandler(self)
        self.cb_keyboard.connect()

        # Class update event callback
        def updater(*args, **kwargs):
            self.refresh()
        callback = MplCallback(registry=self.callbacks_common,
                               event='spy_classes_modified',
                               callback=updater)
        callback.connect()
        self.cb_classes_modified = callback

        if settings.imshow_enable_rectangle_selector is False:
            return
        try:
            from matplotlib.widgets import RectangleSelector
            self.selector = RectangleSelector(self.axes,
                                              self._select_rectangle,
                                              button=1,
                                              useblit=True,
                                              spancoords='data',
                                              props=\
                                                  self.selector_rectprops,
                                              state_modifier_keys=\
                                                  {'square': None,
                                                   'center': None})
            self.selector.set_active(False)
        except:
            raise
            self.selector = None
            msg = 'Failed to create RectangleSelector object. Interactive ' \
              'pixel class labeling will be unavailable.'
            warnings.warn(msg)
            pass

    def label_region(self, rectangle, class_id):
        '''Assigns all pixels in the rectangle to the specified class.

        Arguments:

            `rectangle` (4-tuple of integers):

                Tuple or list defining the rectangle bounds. Should have the
                form (row_start, row_stop, col_start, col_stop), where the
                stop indices are not included (i.e., the effect is
                `classes[row_start:row_stop, col_start:col_stop] = id`.

            class_id (integer >= 0):

                The class to which pixels will be assigned.

        Returns the number of pixels reassigned (the number of pixels in the
        rectangle whose class has *changed* to `class_id`.
        '''
        show_classes = self.classes is None
        if show_classes:
            self.set_classes(np.zeros(self.data_rgb.shape[:2], dtype=np.int16))
        r = rectangle
        n = np.sum(self.classes[r[0]:r[1], r[2]:r[3]] != class_id)
        if n > 0:
            self.classes[r[0]:r[1], r[2]:r[3]] = class_id
            if show_classes:
                self.show_classes()
                self.set_display_mode('overlay')
            event = SpyMplEvent('spy_classes_modified')
            event.classes = self.classes
            event.nchanged = n
            self.callbacks_common.process('spy_classes_modified', event)
            # Make selection rectangle go away.
            self.selector.set_visible(False)
            self.refresh()
            return n
        return 0

    def _select_rectangle(self, event1, event2):
        if event1.inaxes is not self.axes or event2.inaxes is not self.axes:
            self.selection = None
            return
        (r1, c1) = xy_to_rowcol(event1.xdata, event1.ydata)
        (r2, c2) = xy_to_rowcol(event2.xdata, event2.ydata)
        (r1, r2) = sorted([r1, r2])
        (c1, c2) = sorted([c1, c2])
        if (r2 < 0) or (r1 >= self._image_shape[0]) or \
          (c2 < 0) or (c1 >= self._image_shape[1]):
            self.selection = None
            return
        r1 = max(r1, 0)
        r2 = min(r2, self._image_shape[0] - 1)
        c1 = max(c1, 0)
        c2 = min(c2, self._image_shape[1] - 1)
        print('Selected region: [%d: %d, %d: %d]' % (r1, r2 + 1, c1, c2 + 1))
        self.selection = [r1, r2 + 1, c1, c2 + 1]
        self.selector.set_active(False)
        # Make the rectangle display until at least the next event
        self.selector.set_visible(True)
        self.selector.update()

    def _guess_mode(self):
        '''Select an appropriate display mode, based on current data.'''
        if self.data_rgb is not None:
            self.set_display_mode('data')
        elif self.classes is not None:
            self.set_display_mode('classes')
        else:
            raise Exception('Unable to display image: no data set.')

    def show_data(self):
        '''Show the image data.'''
        import matplotlib.pyplot as plt
        if self.data_axes is not None:
            msg = 'ImageView.show_data should only be called once.'
            warnings.warn(UserWarning(msg))
            return
        elif self.data_rgb is None:
            raise Exception('Unable to display data: data array not set.')
        if self.axes is not None:
            # A figure has already been created for the view. Make it current.
            plt.figure(self.axes.figure.number)
        self.imshow_data_kwargs['interpolation'] = self._interpolation
        self.data_axes = plt.imshow(self.data_rgb, **self.imshow_data_kwargs)
        if self.axes is None:
            self.axes = self.data_axes.axes

    def show_classes(self):
        '''Show the class values.'''
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, NoNorm

        if self.class_axes is not None:
            msg = 'ImageView.show_classes should only be called once.'
            warnings.warn(UserWarning(msg))
            return
        elif self.classes is None:
            raise Exception('Unable to display classes: class array not set.')

        cm = ListedColormap(np.array(self.class_colors) / 255.)
        self._update_class_rgb()
        kwargs = self.imshow_class_kwargs.copy()
        kwargs.update({'cmap': cm, 'norm': NoNorm(),
                       'interpolation': self._interpolation})
        if self.axes is not None:
            # A figure has already been created for the view. Make it current.
            plt.figure(self.axes.figure.number)
        self.class_axes = plt.imshow(self.class_rgb, **kwargs)
        if self.axes is None:
            self.axes = self.class_axes.axes
        self.class_axes.set_zorder(1)
        if self.display_mode == 'overlay':
            self.class_axes.set_alpha(self._class_alpha)
        else:
            self.class_axes.set_alpha(1)
        # self.class_axes.axes.set_axis_bgcolor('black')

    def refresh(self):
        '''Updates the displayed data (if it has been shown).'''
        if self.is_shown:
            self._update_class_rgb()
            if self.class_axes is not None:
                self.class_axes.set_data(self.class_rgb)
                self.class_axes.set_interpolation(self._interpolation)
            elif self.display_mode in ('classes', 'overlay'):
                self.show_classes()
            if self.data_axes is not None:
                self.data_axes.set_data(self.data_rgb)
                self.data_axes.set_interpolation(self._interpolation)
            elif self.display_mode in ('data', 'overlay'):
                self.show_data()
            self.axes.figure.canvas.draw()

    def _update_class_rgb(self):
        if self.display_mode == 'overlay':
            self.class_rgb = np.ma.array(self.classes, mask=(self.classes == 0))
        else:
            self.class_rgb = np.array(self.classes)

    def set_display_mode(self, mode):
        '''`mode` must be one of ("data", "classes", "overlay").'''
        if mode not in ('data', 'classes', 'overlay'):
            raise ValueError('Invalid display mode: ' + repr(mode))
        self.display_mode = mode

        show_data = mode in ('data', 'overlay')
        if self.data_axes is not None:
            self.data_axes.set_visible(show_data)

        show_classes = mode in ('classes', 'overlay')
        if self.classes is not None and self.class_axes is None:
            # Class data values were just set
            self.show_classes()
        if self.class_axes is not None:
            self.class_axes.set_visible(show_classes)
            if mode == 'classes':
                self.class_axes.set_alpha(1)
            else:
                self.class_axes.set_alpha(self._class_alpha)
        self.refresh()

    @property
    def class_alpha(self):
        '''alpha transparency for the class overlay.'''
        return self._class_alpha

    @class_alpha.setter
    def class_alpha(self, alpha):
        if alpha < 0 or alpha > 1:
            raise ValueError('Alpha value must be in range [0, 1].')
        self._class_alpha = alpha
        if self.class_axes is not None:
            self.class_axes.set_alpha(alpha)
        if self.is_shown:
            self.refresh()

    @property
    def interpolation(self):
        '''matplotlib pixel interpolation to use in the image display.'''
        return self._interpolation

    @interpolation.setter
    def interpolation(self, interpolation):
        if interpolation == self._interpolation:
            return
        self._interpolation = interpolation
        if not self.is_shown:
            return
        if self.data_axes is not None:
            self.data_axes.set_interpolation(interpolation)
        if self.class_axes is not None:
            self.class_axes.set_interpolation(interpolation)
        self.refresh()

    def set_title(self, s):
        if self.is_shown:
            self.axes.set_title(s)
            self.refresh()

    def open_zoom(self, center=None, size=None):
        '''Opens a separate window with a zoomed view.
        If a ctrl-lclick event occurs in the original view, the zoomed window
        will pan to the location of the click event.

        Arguments:

            `center` (two-tuple of int):

                Initial (row, col) of the zoomed view.

            `size` (int):

                Width and height (in source image pixels) of the initial
                zoomed view.

        Returns:

        A new ImageView object for the zoomed view.
        '''
        from spectral import settings
        import matplotlib.pyplot as plt
        if size is None:
            size = settings.imshow_zoom_pixel_width
        (nrows, ncols) = self._image_shape
        fig_kwargs = {}
        if settings.imshow_zoom_figure_width is not None:
            width = settings.imshow_zoom_figure_width
            fig_kwargs['figsize'] = (width, width)
        fig = plt.figure(**fig_kwargs)

        view = ImageView(source=self.source)
        view.set_data(self.data, self.bands, **self.rgb_kwargs)
        view.set_classes(self.classes, self.class_colors)
        view.imshow_data_kwargs = self.imshow_data_kwargs.copy()
        kwargs = {'extent': (-0.5, ncols - 0.5, nrows - 0.5, -0.5)}
        view.imshow_data_kwargs.update(kwargs)
        view.imshow_class_kwargs = self.imshow_class_kwargs.copy()
        view.imshow_class_kwargs.update(kwargs)
        view.callbacks_common = self.callbacks_common
        view.spectrum_plot_fig_id = self.spectrum_plot_fig_id
        view.show(fignum=fig.number, mode=self.display_mode)
        view.axes.set_xlim(0, size)
        view.axes.set_ylim(size, 0)
        view.interpolation = 'nearest'
        if center is not None:
            view.pan_to(*center)
        view.cb_parent_pan = ParentViewPanCallback(view, self)
        view.cb_parent_pan.connect()
        return view

    def pan_to(self, row, col):
        '''Centers view on pixel coordinate (row, col).'''
        if self.axes is None:
            raise Exception('Cannot pan image until it is shown.')
        (xmin, xmax) = self.axes.get_xlim()
        (ymin, ymax) = self.axes.get_ylim()
        xrange_2 = (xmax - xmin) / 2.0
        yrange_2 = (ymax - ymin) / 2.0
        self.axes.set_xlim(col - xrange_2, col + xrange_2)
        self.axes.set_ylim(row - yrange_2, row + yrange_2)
        self.axes.figure.canvas.draw()

    def zoom(self, scale):
        '''Zooms view in/out (`scale` > 1 zooms in).'''
        (xmin, xmax) = self.axes.get_xlim()
        (ymin, ymax) = self.axes.get_ylim()
        x = (xmin + xmax) / 2.0
        y = (ymin + ymax) / 2.0
        dx = (xmax - xmin) / 2.0 / scale
        dy = (ymax - ymin) / 2.0 / scale

        self.axes.set_xlim(x - dx, x + dx)
        self.axes.set_ylim(y - dy, y + dy)
        self.refresh()

    def format_coord(self, x, y):
        '''Formats pixel coordinate string displayed in the window.'''
        (nrows, ncols) = self._image_shape
        if x < -0.5 or x > ncols - 0.5 or y < -0.5 or y > nrows - 0.5:
            return ""
        (r, c) = xy_to_rowcol(x, y)
        s = 'pixel=[%d,%d]' % (r, c)
        if self.classes is not None:
            try:
                s += ' class=%d' % self.classes[r, c]
            except:
                pass
        return s

    def __str__(self):
        meta = self.data_rgb_meta
        s = 'ImageView object:\n'
        if 'bands' in meta:
            s += '  {0:<20}:  {1}\n'.format("Display bands", meta['bands'])
        if self.interpolation is None:
            interp = "<default>"
        else:
            interp = self.interpolation
        s += '  {0:<20}:  {1}\n'.format("Interpolation", interp)
        if 'rgb range' in meta:
            s += '  {0:<20}:\n'.format("RGB data limits")
            for (c, r) in zip('RGB', meta['rgb range']):
                s += '    {0}: {1}\n'.format(c, str(r))
        return s

    def __repr__(self):
        return str(self)


def imshow(data=None, bands=None, classes=None, source=None, colors=None,
           figsize=None, fignum=None, title=None, **kwargs):
    '''A wrapper around matplotlib's imshow for multi-band images.

    Arguments:

        `data` (SpyFile or ndarray):

            Can have shape (R, C) or (R, C, B).

        `bands` (tuple of integers, optional)

            If `bands` has 3 values, the bands specified are extracted from
            `data` to be plotted as the red, green, and blue colors,
            respectively. If it contains a single value, then a single band
            will be extracted from the image.

        `classes` (ndarray of integers):

            An array of integer-valued class labels with shape (R, C). If
            the `data` argument is provided, the shape must match the first
            two dimensions of `data`. The returned `ImageView` object will use
            a copy of this array. To access class values that were altered
            after calling `imshow`, access the `classes` attribute of the
            returned `ImageView` object.

        `source` (optional, SpyImage or ndarray):

            Object used for accessing image source data. If this argument is
            not provided, events such as double-clicking will have no effect
            (i.e., a spectral plot will not be created).

        `colors` (optional, array of ints):

            Custom colors to be used for class image view. If provided, this
            argument should be an array of 3-element arrays, each of which
            specifies an RGB triplet with integer color components in the
            range [0, 256).

        `figsize` (optional, 2-tuple of scalar):

            Specifies the width and height (in inches) of the figure window
            to be created. If this value is not provided, the value specified
            in `spectral.settings.imshow_figure_size` will be used.

        `fignum`  (optional, integer):

            Specifies the figure number of an existing matplotlib figure. If
            this argument is None, a new figure will be created.

        `title` (str):

            The title to be displayed above the image.

    Keywords:

        Keywords accepted by :func:`~spectral.graphics.graphics.get_rgb` or
        :func:`matplotlib.imshow` will be passed on to the appropriate
        function.

    This function defaults the color scale (imshow's "cmap" keyword) to
    "gray". To use imshow's default color scale, call this function with
    keyword `cmap=None`.

    Returns:

        An `ImageView` object, which can be subsequently used to refine the
        image display.

    See :class:`~spectral.graphics.spypylab.ImageView` for additional details.

    Examples:

    Show a true color image of a hyperspectral image:

        >>> data = open_image('92AV3C.lan').load()
        >>> view = imshow(data, bands=(30, 20, 10))

    Show ground truth in a separate window:

        >>> classes = open_image('92AV3GT.GIS').read_band(0)
        >>> cview = imshow(classes=classes)

    Overlay ground truth data on the data display:

        >>> view.set_classes(classes)
        >>> view.set_display_mode('overlay')

    Show RX anomaly detector results in the view and a zoom window showing
    true color data:

        >>> x = rx(data)
        >>> zoom = view.open_zoom()
        >>> view.set_data(x)

    Note that pressing ctrl-lclick with the mouse in the main window will
    cause the zoom window to pan to the clicked location.

    Opening zoom windows, changing display modes, and other functions can
    also be achieved via keys mapped directly to the displayed image. Press
    "h" with focus on the displayed image to print a summary of mouse/
    keyboard commands accepted by the display.
    '''
    import matplotlib.pyplot as plt

    set_mpl_interactive()

    view = ImageView()
    if data is not None:
        view.set_data(data, bands, **kwargs)
    if classes is not None:
        view.set_classes(classes, colors, **kwargs)
    if source is not None:
        view.set_source(source)
    elif data is not None and len(data.shape) == 3 and data.shape[2] > 3:
        view.set_source(data)
    if fignum is not None or figsize is not None:
        fig = plt.figure(num=fignum, figsize=figsize)
        view.show(fignum=fig.number)
    else:
        view.show()

    if title is not None:
        view.set_title(title)
    return view


def plot(data, source=None):
    '''
    Creates an x-y plot.

    USAGE: plot(data)

    If data is a vector, all the values in data will be drawn in a
    single series. If data is a 2D array, each column of data will
    be drawn as a separate series.
    '''
    import matplotlib.pyplot as plt
    import spectral

    set_mpl_interactive()

    if source is not None and hasattr(source, 'bands') and \
       source.bands.centers is not None:
        xvals = source.bands.centers
    else:
        xvals = list(range(data.shape[-1]))

    if data.ndim == 1:
        data = data[np.newaxis, :]
    data = data.reshape(-1, data.shape[-1])
    if source is not None and hasattr(source, 'metadata') and \
       'bbl' in source.metadata:
        # Do not plot bad bands
        data = np.array(data)
        data[:, np.array(source.metadata['bbl']) == 0] = None
    for x in data:
        p = plt.plot(xvals, x)
    spectral._xyplot = p
    plt.grid(1)
    if source is not None and hasattr(source, 'bands'):
        if source.bands.band_quantity is not None:
            xlabel = source.bands.band_quantity
        else:
            xlabel = ''
        if source.bands.band_unit is not None:
            if len(xlabel) > 0:
                xlabel += ' (%s)' % source.bands.band_unit
            else:
                xlabel = str(source.bands.band_unit)
        plt.xlabel(xlabel)
    return p


def set_mpl_interactive():
    '''Ensure matplotlib is in interactive mode.'''
    import matplotlib.pyplot as plt

    if not plt.isinteractive():
        plt.interactive(True)
