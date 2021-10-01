'''
Code to display N-dimensional data sets in 3D using OpenGL.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import math
import numpy as np
import os
from pprint import pprint
import random
import time

try:
    import wx
    from wx import glcanvas
except ImportError:
    raise ImportError("Required dependency wx.glcanvas not present")

from .. import settings
from ..config import spy_colors
from .colorscale import ColorScale
from .spypylab import ImageView, MplCallback, SpyMplEvent
from .graphics import WindowProxy

DEFAULT_WIN_SIZE = (500, 500)           # Default dimensions of image frame


def rtp_to_xyz(r, theta, phi):
    '''Convert spherical polar coordinates to Cartesian'''
    theta *= math.pi / 180.0
    phi *= math.pi / 180.0
    s = r * math.sin(theta)
    return [s * math.cos(phi), s * math.sin(phi), r * math.cos(theta)]


def xyz_to_rtp(x, y, z):
    '''Convert Cartesian coordinates to Spherical Polar.'''
    r = math.sqrt(x * x + y * y + z * z)
    rho = math.sqrt(x * x + y * y)
    phi = math.asin(y / rho) * 180. / math.pi
    if x < 0.0:
        phi += 180
    theta = math.acos(z / r) * 180. / math.pi
    return [r, theta, phi]

(DOWN, UP) = (1, 0)


class MouseHandler:
    '''A class to enable rotate/zoom functions in an OpenGL window.'''
    MAX_BUTTONS = 10

    def __init__(self, window):
        self.window = window
        self.position = None
        self.event_position = None
        self.left = UP
        self.right = UP
        self.middle = UP
        self.mode = 'DEFAULT'

    def left_down(self, event):
        self.position = (event.X, event.Y)
        self.left = DOWN
        if self.mode == 'DEFAULT':
            if wx.GetKeyState(wx.WXK_CONTROL) and wx.GetKeyState(wx.WXK_SHIFT):
                # Display the row/col and class of the selected pixel.
                (x, y) = self.position
                cmd = lambda: self.window.get_pixel_info(
                    x, self.window.size[1] - y)
                self.window.add_display_command(cmd)
                self.window.canvas.SetCurrent(self.window.canvas.context)
                self.window.canvas.Refresh()
            elif wx.GetKeyState(wx.WXK_SHIFT):
                # Switch to box selection mode.
                print('IN BOX SELECTION MODE.')
                self.mode = 'BOX_SELECT'
            elif wx.GetKeyState(wx.WXK_CONTROL):
                # Switch to zoom mode.
                self.mode = 'ZOOMING'
        self.event_position = (event.X, event.Y)
        event.Skip()

    def left_up(self, event):
        self.position = (event.X, event.Y)
        self.left = UP
        if self.mode == 'BOX_SELECT':
            self.update_box_coordinates()
            # Box selection ends when the button is released.
            if wx.GetKeyState(wx.WXK_SHIFT):
                print('BOX HAS BEEN SELECTED.')
                self.mode = 'DEFAULT'
            else:
                # Shift key was released before box selection completed.
                print('BOX SELECTION CANCELLED.')
                self.window._selection_box = None
            self.window.canvas.SetCurrent(self.window.canvas.context)
            self.window.canvas.Refresh()
        elif self.mode == 'ZOOMING':
            self.mode = 'DEFAULT'
        self.event_position = (event.X, event.Y)
        event.Skip()

    def motion(self, event):
        '''Handles panning & zooming for mouse click+drag events.'''
        if DOWN not in (self.left, self.right):
            return
        #print 'Mouse movement:', x, y
        (w, h) = self.window.size
        dx = event.X - self.position[0]
        dy = event.Y - self.position[1]
        if self.mode == 'DEFAULT':
            if self.left == DOWN and not self.window.mouse_panning:
                # Mouse movement creates a rotation about the target position
                xangle = 2.0 * self.window.fovy * float(dx) / h
                yangle = 2.0 * self.window.fovy * float(dy) / h
                rtp = self.window.camera_pos_rtp
                rtp[1] = min(max(rtp[1] - yangle, 0.05), 179.95)
                self.window.camera_pos_rtp[2] -= xangle
            elif self.left == DOWN:
                # Mouse movement pans target position in the plane of window
                camera_pos = np.array(rtp_to_xyz(*self.window.camera_pos_rtp))
                view_vec = -np.array(rtp_to_xyz(*self.window.camera_pos_rtp))
                zhat = np.array([0.0, 0.0, 1.0])
                right = -np.cross(zhat, view_vec)
                right /= np.sum(np.square(right))
                up = np.cross(right, view_vec)
                up /= np.sum(np.square(up))
                dr = right * (4.0 * dx / w)
                du = up * (4.0 * dy / h)
                self.window.target_pos += du - dr
        elif self.mode == 'ZOOMING':
            # Mouse movement zooms in/out relative to target position
            if dx != 0.0:
                self.window.camera_pos_rtp[0] *= (float(w - dx) / w)
        elif self.mode == 'BOX_SELECT':
            self.update_box_coordinates()
        self.position = (event.X, event.Y)
        self.window.Refresh()
        event.Skip()

    def update_box_coordinates(self):
        xmin = min(self.event_position[0], self.position[0])
        xmax = max(self.event_position[0], self.position[0])
        ymin = min(self.event_position[1], self.position[1])
        ymax = max(self.event_position[1], self.position[1])
        R = self.window.size[1]
        self.window._selection_box = (xmin, R - ymax, xmax, R - ymin)

class MouseMenu(wx.Menu):
    '''Right-click menu for reassigning points to different classes.'''
    ids = []
    def __init__(self, window):
        super(MouseMenu, self).__init__(title='Assign to class')
        self.window = window
        self.id_classes = {}
        while len(self.ids) < self.window.max_menu_class + 1:
            self.ids.append( wx.NewId())
        for i in range(self.window.max_menu_class + 1):
            id = self.ids[i]
            self.id_classes[id] = i
            print('(id, i) =', (id, i))
            mi = wx.MenuItem(self, id, str(i))
            self.AppendItem(mi)
            self.Bind(wx.EVT_MENU, self.reassign_points, mi)

    def reassign_points(self, event):
        i = self.id_classes[event.GetId()]
        self.window.post_reassign_selection(i)

# Multipliers for projecting data into each 3D octant
octant_coeffs = np.array([
    [1, 1, 1],
    [-1, 1, 1],
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, -1],
    [-1, 1, -1],
    [-1, -1, -1],
    [1, -1, -1]], float)


def create_mirrored_octants(feature_indices):
    '''Takes a list of 6 integers and returns 8 lists of feature index
    triplets. The 6 indices passed each specify a feature to be associatd with
    a semi-axis in the 3D display.  Each of the 8 returned triplets specifies
    the 3 features associated with  particular octant, starting with the
    positive x,y,z octant, proceding counterclockwise around the z-axis then
    similarly for the negative half of the z-axis.
    '''
    f = feature_indices
    octants = [
        [f[0], f[1], f[2]],
        [f[3], f[1], f[2]],
        [f[3], f[4], f[2]],
        [f[0], f[4], f[2]],
        [f[0], f[1], f[5]],
        [f[3], f[1], f[5]],
        [f[3], f[4], f[5]],
        [f[0], f[4], f[5]]]
    return octants


def random_subset(sequence, nsamples):
    '''Returns a list of `nsamples` unique random elements from `sequence`.'''
    if len(sequence) < nsamples:
        raise Exception('Sequence in random_triplet must have at least ' +
                        '3 elements.')
    triplet = [random.choice(sequence) for i in range(nsamples)]
    while len(set(triplet)) != nsamples:
        triplet = [random.choice(sequence) for i in range(nsamples)]
    return triplet


class NDWindowProxy(WindowProxy):
    '''A proxy class to retrieve data from an NDWindow.
    An instance contains the following members:

        `classes` (ndarray):

            The current class labels associated with the NDWindow data.

        `set_features` ((list, string)):

            List of features and display mode (see set_features doc string.)
    '''
    def __init__(self, window):
        WindowProxy.__init__(self, window)
        self._classes = window.classes

    @property
    def classes(self):
        '''Returns the current class labels associated with data points.'''
        return self._classes

    def set_features(self, *args, **kwargs):
        '''Specifies which features to display in the 3D window.

        Arguments:

        `features` (list or list of integer lists):

            This keyword specifies which bands/features from `data` should be
            displayed in the 3D window. It must be defined as one of the
            following:

            #. If `mode` is set to "single" (the default), then `features`
               must be a length-3 list of integer feature IDs. In this case,
               the data points will be displayed in the positive x,y,z octant
               using features associated with the 3 integers.

            #. If `mode` is set to "mirrored", then `features` must be a
               length-6 list of integer feature IDs. In this case, each
               integer specifies a single feature index to be associated with
               the coordinate semi-axes x, y, z, -x, -y, and -z (in that
               order). Each octant will display data points using the features
               associated with the 3 semi-axes for that octant.

            #. If `mode` is set to "independent", then `features` must be a
               length-8 list of length-3 lists of integers. In this case, each
               length-3 list specfies the features to be displayed in a single
               octants (the same semi-axis can be associated with different
               features in different octants).  Octants are ordered starting
               with the postive x,y,z octant and procede counterclockwise
               around the z-axis, then procede similarly around the negative
               half of the z-axis.  An octant triplet can be specified as None
               instead of a list, in which case nothing will be rendered in
               that octant.

        `mode` (string, default="single")

            The display mode for the 3D octants.  This value must be "single",
            "mirrored", or "independent".
        '''

        if not isinstance(self._window, wx.Frame):
            raise Exception('The window no longer exists.')
        self._window.set_features(*args, **kwargs)

    def view_class_image(self, *args, **kwargs):
        '''Show a dynamically updated view of image class values.

        The class IDs displayed are those currently associated with the ND
        window. `args` and `kwargs` are additional arguments passed on to the
        `ImageView` constructor. Return value is the ImageView object.
        '''
        return self._window.view_class_image(*args, **kwargs)

class NDWindow(wx.Frame):
    '''A widow class for displaying N-dimensional data points.'''

    def __init__(self, data, parent, id, *args, **kwargs):
        global DEFAULT_WIN_SIZE
        self.kwargs = kwargs
        self.size = kwargs.get('size', DEFAULT_WIN_SIZE)
        self.title = kwargs.get('title', 'ND Window')

        #
        # Forcing a specific style on the window.
        #   Should this include styles passed?
        style = wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE
        super(NDWindow, self).__init__(parent, id, self.title,
                                       wx.DefaultPosition,
                                       wx.Size(*self.size),
                                       style,
                                       self.title)

        self.gl_initialized = False
        attribs = (glcanvas.WX_GL_RGBA,
                   glcanvas.WX_GL_DOUBLEBUFFER,
                   glcanvas.WX_GL_DEPTH_SIZE, settings.WX_GL_DEPTH_SIZE)
        self.canvas = glcanvas.GLCanvas(self, attribList=attribs)
        self.canvas.context = wx.glcanvas.GLContext(self.canvas)

        self._have_glut = False
        self.clear_color = (0, 0, 0, 0)
        self.show_axes_tf = True
        self.point_size = 1.0
        self._show_unassigned = True
        self._refresh_display_lists = False
        self._click_tolerance = 1
        self._display_commands = []
        self._selection_box = None
        self._rgba_indices = None
        self.mouse_panning = False
        self.win_pos = (100, 100)
        self.fovy = 60.
        self.znear = 0.1
        self.zfar = 10.0
        self.target_pos = [0.0, 0.0, 0.0]
        self.camera_pos_rtp = [7.0, 45.0, 30.0]
        self.up = [0.0, 0.0, 1.0]

        self.quadrant_mode = None
        self.mouse_handler = MouseHandler(self)

        # Set the event handlers.
        self.canvas.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)
        self.Bind(wx.EVT_SIZE, self.on_resize)
        self.canvas.Bind(wx.EVT_PAINT, self.on_paint)
        self.canvas.Bind(wx.EVT_LEFT_DOWN, self.mouse_handler.left_down)
        self.canvas.Bind(wx.EVT_LEFT_UP, self.mouse_handler.left_up)
        self.canvas.Bind(wx.EVT_MOTION, self.mouse_handler.motion)
        self.canvas.Bind(wx.EVT_CHAR, self.on_char)
        self.canvas.Bind(wx.EVT_RIGHT_DOWN, self.right_click)
        self.canvas.Bind(wx.EVT_CLOSE, self.on_event_close)

        self.data = data
        self.classes = kwargs.get('classes',
                                  np.zeros(data.shape[:-1], int))
        self.features = kwargs.get('features', list(range(6)))
        self.labels = kwargs.get('labels', list(range(data.shape[-1])))
        self.max_menu_class = int(np.max(self.classes.ravel() + 1))

        from matplotlib.cbook import CallbackRegistry
        self.callbacks = CallbackRegistry()


    def on_event_close(self, event=None):
        pass

    def right_click(self, event):
        self.canvas.SetCurrent(self.canvas.context)
        self.canvas.PopupMenu(MouseMenu(self), event.GetPosition())

    def add_display_command(self, cmd):
        '''Adds a command to be called next time `display` is run.'''
        self._display_commands.append(cmd)

    def reset_view_geometry(self):
        '''Sets viewing geometry to the default view.'''
        # All grid points will be adjusted to the range [0,1] so this
        # is a reasonable center coordinate for the scene
        self.target_pos = np.array([0.0, 0.0, 0.0])

        # Specify the camera location in spherical polar coordinates relative
        # to target_pos.
        self.camera_pos_rtp = [2.5, 45.0, 30.0]

    def set_data(self, data, **kwargs):
        '''Associates N-D point data with the window.
        ARGUMENTS:
            data (numpy.ndarray):
                An RxCxB array of data points to display.
        KEYWORD ARGUMENTS:
            classes (numpy.ndarray):
                An RxC array of integer class labels (zeros means unassigned).
            features (list):
                Indices of feautures to display in the octant (see
                NDWindow.set_octant_display_features for description).
        '''
        import OpenGL.GL as gl
        try:
            from OpenGL.GL import glGetIntegerv
        except:
            from OpenGL.GL.glget import glGetIntegerv

        classes = kwargs.get('classes', None)
        features = kwargs.get('features', list(range(6)))
        if self.data.shape[2] < 6:
            features = features[:3]
            self.quadrant_mode == 'single'

        # Scale the data set to span an octant

        data2d = np.array(data.reshape((-1, data.shape[-1])))
        mins = np.min(data2d, axis=0)
        maxes = np.max(data2d, axis=0)
        denom = (maxes - mins).astype(float)
        denom = np.where(denom > 0, denom, 1.0)
        self.data = (data2d - mins) / denom
        self.data.shape = data.shape

        self.palette = spy_colors.astype(float) / 255.
        self.palette[0] = np.array([1.0, 1.0, 1.0])
        self.colors = self.palette[self.classes.ravel()].reshape(
            self.data.shape[:2] + (3,))
        self.colors = (self.colors * 255).astype('uint8')
        colors = np.ones((self.colors.shape[:-1]) + (4,), 'uint8')
        colors[:, :, :-1] = self.colors
        self.colors = colors
        self._refresh_display_lists = True
        self.set_octant_display_features(features)

        # Determine the bit masks to use when using RGBA components for
        # identifying pixel IDs.
        components = [gl.GL_RED_BITS, gl.GL_GREEN_BITS,
                      gl.GL_GREEN_BITS, gl.GL_ALPHA_BITS]
        self._rgba_bits = [min(8, glGetIntegerv(i)) for i in components]
        self._low_bits = [min(8, 8 - self._rgba_bits[i]) for i in range(4)]
        self._rgba_masks = \
            [(2**self._rgba_bits[i] - 1) << (8 - self._rgba_bits[i])
             for i in range(4)]

        # Determine how many times the scene will need to be rendered in the
        # background to extract the pixel's row/col index.

        N = self.data.shape[0] * self.data.shape[1]
        if N > 2**sum(self._rgba_bits):
            raise Exception('Insufficient color bits (%d) for N-D window display'
                            % sum(self._rgba_bits))
        self.reset_view_geometry()

    def set_octant_display_features(self, features):
        '''Specifies features to be displayed in each 3-D coordinate octant.
        `features` can be any of the following:
        A length-3 list of integer feature IDs:
            In this case, the data points will be displayed in the positive
            x,y,z octant using features associated with the 3 integers.
        A length-6 list if integer feature IDs:
            In this case, each integer specifies a single feature index to be
            associated with the coordinate semi-axes x, y, z, -x, -y, and -z
            (in that order).  Each octant will display data points using the
            features associated with the 3 semi-axes for that octant.
        A length-8 list of length-3 lists of integers:
            In this case, each length-3 list specfies the features to be
            displayed in a single octants (the same semi-axis can be associated
            with different features in different octants).  Octants are ordered
            starting with the postive x,y,z octant and procede counterclockwise
            around the z-axis, then procede similarly around the negative half
            of the z-axis.  An octant triplet can be specified as None instead
            of a list, in which case nothing will be rendered in that octant.
        '''
        if features is None:
            features = list(range(6))
        if len(features) == 3:
            self.octant_features = [features] + [None] * 7
            new_quadrant_mode = 'single'
            self.target_pos = np.array([0.5, 0.5, 0.5])
        elif len(features) == 6:
            self.octant_features = create_mirrored_octants(features)
            new_quadrant_mode = 'mirrored'
            self.target_pos = np.array([0.0, 0.0, 0.0])
        else:
            self.octant_features = features
            new_quadrant_mode = 'independent'
            self.target_pos = np.array([0.0, 0.0, 0.0])
        if new_quadrant_mode != self.quadrant_mode:
            print('Setting quadrant display mode to %s.' % new_quadrant_mode)
            self.quadrant_mode = new_quadrant_mode
        self._refresh_display_lists = True

    def create_display_lists(self, npass=-1, **kwargs):
        '''Creates or updates the display lists for image data.
        ARGUMENTS:
            `npass` (int):
                When defaulted to -1, the normal image data display lists are
                created.  When >=0, `npass` represents the rendering pass for
                identifying image pixels in the scene by their unique colors.
        KEYWORD ARGS:
            `indices` (list of ints):
                 An optional list of N-D image pixels to display.
        '''
        import OpenGL.GL as gl
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)

        gl.glPointSize(self.point_size)
        gl.glColorPointerub(self.colors)

        (R, C, B) = self.data.shape

        indices = kwargs.get('indices', None)
        if indices is None:
            indices = np.arange(R * C)
            if not self._show_unassigned:
                indices = indices[self.classes.ravel() != 0]
            self._display_indices = indices

        # RGB pixel indices for selecting pixels with the mouse
        gl.glPointSize(self.point_size)
        if npass < 0:
            # Colors are associated with image pixel classes.
            gl.glColorPointerub(self.colors)
        else:
            if self._rgba_indices is None:
                # Generate unique colors that correspond to each pixel's ID
                # so that the color can be used to identify the pixel.
                color_indices = np.arange(R * C)
                rgba = np.zeros((len(color_indices), 4), 'uint8')
                for i in range(4):
                    shift = sum(self._rgba_bits[0:i]) - self._low_bits[i]
                    if shift > 0:
                        rgba[:, i] = (
                            color_indices >> shift) & self._rgba_masks[i]
                    else:
                        rgba[:, i] = (color_indices << self._low_bits[i]) \
                            & self._rgba_masks[i]
                self._rgba_indices = rgba
            gl.glColorPointerub(self._rgba_indices)

        # Generate a display list for each octant of the 3-D window.

        for (i, octant) in enumerate(self.octant_features):
            if octant is not None:
                data = np.take(self.data, octant, axis=2).reshape((-1, 3))
                data *= octant_coeffs[i]
                gl.glVertexPointerf(data)
                gl.glNewList(self.gllist_id + i + 1, gl.GL_COMPILE)
                gl.glDrawElementsui(gl.GL_POINTS, indices)
                gl.glEndList()
            else:
                # Create an empty draw list
                gl.glNewList(self.gllist_id + i + 1, gl.GL_COMPILE)
                gl.glEndList()

        self.create_axes_list()
        self._refresh_display_lists = False

    def randomize_features(self):
        '''Randomizes data features displayed using current display mode.'''
        ids = list(range(self.data.shape[2]))
        if self.quadrant_mode == 'single':
            features = random_subset(ids, 3)
        elif self.quadrant_mode == 'mirrored':
            features = random_subset(ids, 6)
        else:
            features = [random_subset(ids, 3) for i in range(8)]
        print('New feature IDs:')
        pprint(np.array(features))
        self.set_octant_display_features(features)

    def set_features(self, features, mode='single'):
        if mode == 'single':
            if len(features) != 3:
                raise Exception(
                    'Expected 3 feature indices for "single" mode.')
        elif mode == 'mirrored':
            if len(features) != 6:
                raise Exception(
                    'Expected 6 feature indices for "mirrored" mode.')
        elif mode == 'independent':
            if len(features) != 8:
                raise Exception('Expected 8 3-tuples of feature indices for'
                                '"independent" mode.')
        else:
            raise Exception('Unrecognized feature mode: %s.' % str(mode))
        print('New feature IDs:')
        pprint(np.array(features))
        self.set_octant_display_features(features)
        self.Refresh()

    def draw_box(self, x0, y0, x1, y1):
        '''Draws a selection box in the 3-D window.
        Coordinates are with respect to the lower left corner of the window.
        '''
        import OpenGL.GL as gl
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0.0, self.size[0],
                   0.0, self.size[1],
                   -0.01, 10.0)

        gl.glLineStipple(1, 0xF00F)
        gl.glEnable(gl.GL_LINE_STIPPLE)
        gl.glLineWidth(1.0)
        gl.glColor3f(1.0, 1.0, 1.0)
        gl.glBegin(gl.GL_LINE_LOOP)
        gl.glVertex3f(x0, y0, 0.0)
        gl.glVertex3f(x1, y0, 0.0)
        gl.glVertex3f(x1, y1, 0.0)
        gl.glVertex3f(x0, y1, 0.0)
        gl.glEnd()
        gl.glDisable(gl.GL_LINE_STIPPLE)
        gl.glFlush()

        self.resize(*self.size)

    def on_paint(self, event):
        '''Renders the entire scene.'''
        import OpenGL.GL as gl
        import OpenGL.GLU as glu

        self.canvas.SetCurrent(self.canvas.context)
        if not self.gl_initialized:
            self.initgl()
            self.gl_initialized = True
            self.print_help()
            self.resize(*self.size)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        while len(self._display_commands) > 0:
            self._display_commands.pop(0)()

        if self._refresh_display_lists:
            self.create_display_lists()

        gl.glPushMatrix()
        # camera_pos_rtp is relative to target position. To get the absolute
        # camera position, we need to add the target position.
        camera_pos_xyz = np.array(rtp_to_xyz(*self.camera_pos_rtp)) \
            + self.target_pos
        glu.gluLookAt(
            *(list(camera_pos_xyz) + list(self.target_pos) + self.up))

        if self.show_axes_tf:
            gl.glCallList(self.gllist_id)

        self.draw_data_set()

        gl.glPopMatrix()
        gl.glFlush()

        if self._selection_box is not None:
            self.draw_box(*self._selection_box)

        self.SwapBuffers()
        event.Skip()

    def post_reassign_selection(self, new_class):
        '''Reassigns pixels in selection box during the next rendering loop.
        ARGUMENT:
            `new_class` (int):
                The class to which the pixels in the box will be assigned.
        '''
        if self._selection_box is None:
            msg = 'Bounding box is not selected. Hold SHIFT and click & ' + \
                  'drag with the left\nmouse button to select a region.'
            print(msg)
            return 0
        self.add_display_command(lambda: self.reassign_selection(new_class))
        self.canvas.Refresh()
        return 0

    def reassign_selection(self, new_class):
        '''Reassigns pixels in the selection box to the specified class.
        This method should only be called from the `display` method. Pixels are
        reassigned by identifying each pixel in the 3D display by their unique
        color, then reassigning them. Since pixels can block others in the
        z-buffer, this method iteratively reassigns pixels by removing any
        reassigned pixels from the display list, then reassigning again,
        repeating until there are no more pixels in the selction box.
        '''
        nreassigned_tot = 0
        i = 1
        print('Reassigning points', end=' ')
        while True:
            indices = np.array(self._display_indices)
            classes = np.array(self.classes.ravel()[indices])
            indices = indices[np.where(classes != new_class)]
            ids = self.get_points_in_selection_box(indices=indices)
            cr = self.classes.ravel()
            nreassigned = np.sum(cr[ids] != new_class)
            nreassigned_tot += nreassigned
            cr[ids] = new_class
            new_color = np.zeros(4, 'uint8')
            new_color[:3] = (np.array(self.palette[new_class])
                             * 255).astype('uint8')
            self.colors.reshape((-1, 4))[ids] = new_color
            self.create_display_lists()
            if len(ids) == 0:
                break
#           print 'Pass %d: %d points reassigned to class %d.' \
#                 % (i, nreassigned, new_class)
            print('.', end=' ')
            i += 1
        print('\n%d points were reasssigned to class %d.' \
              % (nreassigned_tot, new_class))
        self._selection_box = None
        if nreassigned_tot > 0 and new_class == self.max_menu_class:
            self.max_menu_class += 1

        if nreassigned_tot > 0:
            event = SpyMplEvent('spy_classes_modified')
            event.classes = self.classes
            event.nchanged = nreassigned_tot
            self.callbacks.process('spy_classes_modified', event)

        return nreassigned_tot

    def get_points_in_selection_box(self, **kwargs):
        '''Returns pixel IDs of all points in the current selection box.
        KEYWORD ARGS:
            `indices` (ndarray of ints):
                An alternate set of N-D image pixels to display.

        Pixels are identified by performing a background rendering loop wherein
        each pixel is rendered with a unique color. Then, glReadPixels is used
        to read colors of pixels in the current selection box.
        '''
        import OpenGL.GL as gl
        indices = kwargs.get('indices', None)
        point_size_temp = self.point_size
        self.point_size = kwargs.get('point_size', 1)

        xsize = self._selection_box[2] - self._selection_box[0] + 1
        ysize = self._selection_box[3] - self._selection_box[1] + 1
        ids = np.zeros(xsize * ysize, int)

        self.create_display_lists(0, indices=indices)
        self.render_rgb_indexed_colors()
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        pixels = gl.glReadPixelsub(self._selection_box[0],
                                   self._selection_box[1],
                                   xsize, ysize, gl.GL_RGBA)
        pixels = np.frombuffer(pixels, dtype=np.uint8).reshape((ysize, xsize, 4))
        for i in range(4):
            component = pixels[:, :, i].reshape((xsize * ysize,)) \
                & self._rgba_masks[i]
            shift = (sum(self._rgba_bits[0:i]) - self._low_bits[i])
            if shift > 0:
                ids += component.astype(int) << shift
            else:
                ids += component.astype(int) >> (-shift)

        points = ids[ids > 0]

        self.point_size = point_size_temp
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self._refresh_display_lists = True

        return points

    def get_pixel_info(self, x, y, **kwargs):
        '''Prints row/col of the pixel at the given raster position.
        ARGUMENTS:
            `x`, `y`: (int):
                The pixel's coordinates relative to the lower left corner.
        '''
        self._selection_box = (x, y, x, y)
        ids = self.get_points_in_selection_box(point_size=self.point_size)
        for id in ids:
            if id > 0:
                rc = self.index_to_image_row_col(id)
                print('Pixel %d %s has class %s.' % (id, rc, self.classes[rc]))
        return

    def render_rgb_indexed_colors(self, **kwargs):
        '''Draws scene in the background buffer to extract mouse click info'''
        import OpenGL.GL as gl
        import OpenGL.GLU as glu
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # camera_pos_rtp is relative to the target position. To get the
        # absolute camera position, we need to add the target position.
        gl.glPushMatrix()
        camera_pos_xyz = np.array(rtp_to_xyz(*self.camera_pos_rtp)) \
            + self.target_pos
        glu.gluLookAt(
            *(list(camera_pos_xyz) + list(self.target_pos) + self.up))
        self.draw_data_set()
        gl.glPopMatrix()
        gl.glFlush()

    def index_to_image_row_col(self, index):
        '''Converts the unraveled pixel ID to row/col of the N-D image.'''
        rowcol = (index // self.data.shape[1], index % self.data.shape[1])
        return rowcol

    def draw_data_set(self):
        '''Draws the N-D data set in the scene.'''
        import OpenGL.GL as gl
        for i in range(1, 9):
            gl.glCallList(self.gllist_id + i)

    def create_axes_list(self):
        '''Creates display lists to render unit length x,y,z axes.'''
        import OpenGL.GL as gl
        gl.glNewList(self.gllist_id, gl.GL_COMPILE)
        gl.glBegin(gl.GL_LINES)
        gl.glColor3f(1.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(1.0, 0.0, 0.0)
        gl.glColor3f(0.0, 1.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 1.0, 0.0)
        gl.glColor3f(-.0, 0.0, 1.0)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 1.0)

        gl.glColor3f(1.0, 1.0, 1.0)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(-1.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, -1.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, -1.0)
        gl.glEnd()

        def label_axis(x, y, z, label):
            gl.glRasterPos3f(x, y, z)
            glut.glutBitmapString(glut.GLUT_BITMAP_HELVETICA_18,
                                  str(label))
        def label_axis_for_feature(x, y, z, feature_ind):
            feature = self.octant_features[feature_ind[0]][feature_ind[1]]
            label_axis(x, y, z, self.labels[feature])

        if self._have_glut:
            try:
                import OpenGL.GLUT as glut
                if bool(glut.glutBitmapString):
                    if self.quadrant_mode == 'independent':
                        label_axis(1.05, 0.0, 0.0, 'x')
                        label_axis(0.0, 1.05, 0.0, 'y')
                        label_axis(0.0, 0.0, 1.05, 'z')
                    elif self.quadrant_mode == 'mirrored':
                        label_axis_for_feature(1.05, 0.0, 0.0, (0, 0))
                        label_axis_for_feature(0.0, 1.05, 0.0, (0, 1))
                        label_axis_for_feature(0.0, 0.0, 1.05, (0, 2))
                        label_axis_for_feature(-1.05, 0.0, 0.0, (6, 0))
                        label_axis_for_feature(0.0, -1.05, 0.0, (6, 1))
                        label_axis_for_feature(0.0, 0.0, -1.05, (6, 2))
                    else:
                        label_axis_for_feature(1.05, 0.0, 0.0, (0, 0))
                        label_axis_for_feature(0.0, 1.05, 0.0, (0, 1))
                        label_axis_for_feature(0.0, 0.0, 1.05, (0, 2))
            except:
                pass
        gl.glEndList()

    def GetGLExtents(self):
        """Get the extents of the OpenGL canvas."""
        return

    def SwapBuffers(self):
        """Swap the OpenGL buffers."""
        self.canvas.SwapBuffers()

    def on_erase_background(self, event):
        """Process the erase background event."""
        pass  # Do nothing, to avoid flashing on MSWin

    def initgl(self):
        '''App-specific initialization for after GLUT has been initialized.'''
        import OpenGL.GL as gl
        self.gllist_id = gl.glGenLists(9)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glDisable(gl.GL_FOG)
        gl.glDisable(gl.GL_COLOR_MATERIAL)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glShadeModel(gl.GL_FLAT)
        self.set_data(self.data, classes=self.classes, features=self.features)

        try:
            import OpenGL.GLUT as glut
            glut.glutInit()
            self._have_glut = True
        except:
            pass

    def on_resize(self, event):
        '''Process the resize event.'''

        # For wx versions 2.9.x, GLCanvas.GetContext() always returns None,
        # whereas 2.8.x will return the context so test for both versions.

        if wx.VERSION >= (2, 9) or self.canvas.GetContext():
            self.canvas.SetCurrent(self.canvas.context)
            # Make sure the frame is shown before calling SetCurrent.
            self.Show()
            size = event.GetSize()
            self.resize(size.width, size.height)
            self.canvas.Refresh(False)
        event.Skip()

    def resize(self, width, height):
        """Reshape the OpenGL viewport based on dimensions of the window."""
        import OpenGL.GL as gl
        import OpenGL.GLU as glu
        self.size = (width, height)
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(self.fovy, float(width) / height,
                           self.znear, self.zfar)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def on_char(self, event):
        '''Callback function for when a keyboard button is pressed.'''
        key = chr(event.GetKeyCode())

        # See `print_help` method for explanation of keybinds.
        if key == 'a':
            self.show_axes_tf = not self.show_axes_tf
        elif key == 'c':
            self.view_class_image()
        elif key == 'd':
            if self.data.shape[2] < 6:
                print('Only single-quadrant mode is supported for %d features.' % \
                      self.data.shape[2])
                return
            if self.quadrant_mode == 'single':
                self.quadrant_mode = 'mirrored'
            elif self.quadrant_mode == 'mirrored':
                self.quadrant_mode = 'independent'
            else:
                self.quadrant_mode = 'single'
            print('Setting quadrant display mode to %s.' % self.quadrant_mode)
            self.randomize_features()
        elif key == 'f':
            self.randomize_features()
        elif key == 'h':
            self.print_help()
        elif key == 'm':
            self.mouse_panning = not self.mouse_panning
        elif key == 'p':
            self.point_size += 1
            self._refresh_display_lists = True
        elif key == 'P':
            self.point_size = max(self.point_size - 1, 1.0)
            self._refresh_display_lists = True
        elif key == 'q':
            self.on_event_close()
            self.Close(True)
        elif key == 'r':
            self.reset_view_geometry()
        elif key == 'u':
            self._show_unassigned = not self._show_unassigned
            print('SHOW UNASSIGNED =', self._show_unassigned)
            self._refresh_display_lists = True
 
        self.canvas.Refresh()

    def update_window_title(self):
        '''Prints current file name and current point color to window title.'''
        from OpenGL.GLUT import glutSetWindowTitle
        s = 'SPy N-D Data Set'
        glutSetWindowTitle(s)

    def get_proxy(self):
        '''Returns a proxy object to access data from the window.'''
        return NDWindowProxy(self)

    def view_class_image(self, *args, **kwargs):
        '''Opens a dynamic raster image of class values.

        The class IDs displayed are those currently associated with the ND
        window. `args` and `kwargs` are additional arguments passed on to the
        `ImageView` constructor. Return value is the ImageView object.
        '''
        view = ImageView(classes=self.classes, *args, **kwargs)
        view.callbacks_common = self.callbacks
        view.show()
        return view

    def print_help(self):
        '''Prints a list of accepted keyboard/mouse inputs.'''
        print('''Mouse functions:
---------------
Left-click & drag       -->     Rotate viewing geometry (or pan)
CTRL+Left-click & drag  -->     Zoom viewing geometry
CTRL+SHIFT+Left-click   -->     Print image row/col and class of selected pixel
SHIFT+Left-click & drag -->     Define selection box in the window
Right-click             -->     Open GLUT menu for pixel reassignment

Keyboard functions:
-------------------
a       -->     Toggle axis display
c       -->     View dynamic raster image of class values
d       -->     Cycle display mode between single-quadrant, mirrored octants,
                and independent octants (display will not change until features
                are randomzed again)
f       -->     Randomize features displayed
h       -->     Print this help message
m       -->     Toggle mouse function between rotate/zoom and pan modes
p/P     -->     Increase/Decrease the size of displayed points
q       -->     Exit the application
r       -->     Reset viewing geometry
u       -->     Toggle display of unassigned points (points with class == 0)
''')


def validate_args(data, *args, **kwargs):
    '''Validates arguments to the `ndwindow` function.'''
    if not isinstance(data, np.ndarray):
        raise TypeError('`data` argument must be a numpy ndarray.')
    if len(data.shape) != 3:
        raise ValueError('`data` argument must have 3 dimensions.')
    if data.shape[2] < 3:
        raise ValueError('`data` argument must have at least 3 values along' +
                         ' third dimension.')
    if 'classes' in kwargs:
        classes = kwargs['classes']
        if classes.shape != data.shape[:2]:
            raise ValueError('`classes` keyword argument shape does not match'
                             ' `data` argument shape.')
    if 'features' in kwargs:
        features = kwargs['features']
        if type(features) not in (list, tuple):
            raise TypeError('`features` keyword must be a list or tuple.')
        if len(features) in (3, 6):
            if max(features) >= data.shape[2]:
                raise ValueError('Feature index exceeds max for data array.')
        elif len(features) == 8:
            for octant in features:
                if type(octant) not in (list, tuple, type(None)):
                    raise TypeError('Each octant in `features` keyword must' +
                                    'be a list/tuple of 3 ints or None.')
                if type(octant) not in (list, tuple) and len(octant) != 3:
                    raise TypeError('Each octant in the `features` keyword ' +
                                    'must be a list/tuple of exactly 3 ints.')
                if max(octant) >= data.shape[2]:
                    raise ValueError(
                        'Feature index exceeds max for data array.')
        else:
            raise ValueError(
                'Invalid number of elements in `features` keyword.')
    if 'size' in kwargs:
        size = kwargs['size']
        if type(size) not in (list, tuple) or len(size) != 2:
            raise ValueError(
                '`size` keyword must be a list/tuple of two ints.')
        for n in size:
            if type(n) != int:
                raise TypeError('`size` keyword must contain two ints.')
            if n < 1:
                raise ValueError('Invalid window size specification.')
    if 'title' in kwargs and type(kwargs['title']) != str:
        raise TypeError('Invalide window title specification.')
