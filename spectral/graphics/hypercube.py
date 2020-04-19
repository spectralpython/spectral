'''
Code for rendering and manipulating hypercubes.
Most users will only need to call the function "hypercube".
'''
# The OpenGL code in this file was adapted from a number of OpenGL demo
# scripts that were created, ported, and adapted by various authors
# including Richard Campbell, John Ferguson, Tony Colston, Tarn Weisner,
# Yan Wong, Greg Landrum, and possibly others.
#
# Source file comments from some of the original files are as follows:
#
#
#------------------------------------------------------------------------
# Ported to PyOpenGL 2.0 by Tarn Weisner Burton 10May2001
#
# This code was created by Richard Campbell '99 (ported to Python/PyOpenGL by
#
# John Ferguson 2000) The port was based on the lesson5 tutorial module by Tony
#
# Colston (tonetheman@hotmail.com). If you've found this code useful, please
#
# let me know (email John Ferguson at hakuin@voicenet.com).
#
# See original source and C based tutorial at http:#nehe.gamedev.net
#------------------------------------------------------------------------
# This file found at:
#   http://lists.wxwidgets.org/archive/wxPython-users/msg11078.html
#
# This includes the two classes wxGLWindow and wxAdvancedGLWindow
# from OpenGL.TK in the PyOpenGL distribution
# ported to wxPython by greg Landrum
# modified by Y. Wong
#------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function, unicode_literals

import math
import numpy as np

try:
    import wx
    from wx import glcanvas
except ImportError:
    raise ImportError("Required dependency wx.glcanvas not present")

from .. import settings
from ..image import Image
from ..io.spyfile import SpyFile
from .colorscale import create_default_color_scale
from .graphics import make_pil_image, SpyWindow

DEFAULT_WIN_SIZE = (500, 500)           # Default dimensions of image frame
DEFAULT_TEXTURE_SIZE = (
    256, 256)       # Default size of textures on cube faces


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

    def left_down(self, event):
        self.event_position = (event.X, event.Y)
        self.position = (event.X, event.Y)
        self.left = DOWN
        event.Skip()

    def left_up(self, event):
        self.position = (event.X, event.Y)
        self.left = UP
        event.Skip()

    def motion(self, event):
        '''Handles panning & zooming for mouse click+drag events.'''
        if DOWN not in (self.left, self.right):
            return
        #print 'Mouse movement:', x, y
        (w, h) = self.window.size
        dx = event.X - self.position[0]
        dy = event.Y - self.position[1]
        if self.left == DOWN:
            if wx.GetKeyState(wx.WXK_CONTROL):
                # Mouse movement zooms in/out relative to target position
                if dx != 0.0:
                    self.window.camera_pos_rtp[0] *= (float(w - dx) / w)
            elif wx.GetKeyState(wx.WXK_SHIFT):
                # Mouse movement pans target position in  plane of the window
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
            else:
                # Mouse movement creates a rotation about the target position
                xangle = 2.0 * self.window.fovy * float(dx) / h
                yangle = 2.0 * self.window.fovy * float(dy) / h
                rtp = self.window.camera_pos_rtp
                rtp[1] = min(max(rtp[1] - yangle, 0.05), 179.95)
                self.window.camera_pos_rtp[2] -= xangle
        self.position = (event.X, event.Y)
        self.window.Refresh()
        event.Skip()

class HypercubeWindow(wx.Frame, SpyWindow):
    """A simple class for using OpenGL with wxPython."""

    def __init__(self, data, parent, id, *args, **kwargs):
        global DEFAULT_WIN_SIZE

        self.kwargs = kwargs
        self.size = kwargs.get('size', DEFAULT_WIN_SIZE)
        self.title = kwargs.get('title', 'Hypercube')

        #
        # Forcing a specific style on the window.
        #   Should this include styles passed?
        style = wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE
        wx.Frame.__init__(self, parent, id, self.title,
                          wx.DefaultPosition,
                          wx.Size(*self.size),
                          style,
                          kwargs.get('name', 'Hypercube'))

        self.gl_initialized = False
        attribs = (glcanvas.WX_GL_RGBA,  # RGBA
                   glcanvas.WX_GL_DOUBLEBUFFER,  # Double Buffered
                   glcanvas.WX_GL_DEPTH_SIZE, settings.WX_GL_DEPTH_SIZE)
        self.canvas = glcanvas.GLCanvas(
            self, attribList=attribs, size=self.size)
        self.canvas.context = wx.glcanvas.GLContext(self.canvas)

        # These members can be modified before calling the show method.
        self.clear_color = tuple(kwargs.get('background', (0., 0., 0.))) \
                           + (1.,)
        self.win_pos = (100, 100)
        self.fovy = 60.
        self.znear = 0.1
        self.zfar = 10.0
        self.target_pos = [0.0, 0.0, 0.0]
        self.camera_pos_rtp = [7.0, 45.0, 30.0]
        self.up = [0.0, 0.0, 1.0]

        self.hsi = data
        self.cubeHeight = 1.0
        self.rotation = [-60, 0, -30]
        self.distance = -5
        self.light = False

        self.texturesLoaded = False
        self.mouse_handler = MouseHandler(self)

        # Set the event handlers.
        self.canvas.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)
        self.canvas.Bind(wx.EVT_SIZE, self.on_resize)
        self.canvas.Bind(wx.EVT_PAINT, self.on_paint)
        self.canvas.Bind(wx.EVT_LEFT_DOWN, self.mouse_handler.left_down)
        self.canvas.Bind(wx.EVT_LEFT_UP, self.mouse_handler.left_up)
        self.canvas.Bind(wx.EVT_MOTION, self.mouse_handler.motion)
        self.canvas.Bind(wx.EVT_CHAR, self.on_char)

    def load_textures(self):
        import OpenGL.GL as gl

        global DEFAULT_TEXTURE_SIZE

        if 'scale' in self.kwargs:
            scale = self.kwargs['scale']
        else:
            scale = create_default_color_scale(256)

        data = self.hsi
        s = data.shape

        # Create image for top of cube
        if 'top' in self.kwargs:
            image = self.kwargs['top']
            if isinstance(image, np.ndarray):
                image = make_pil_image(image)
        else:
            if 'bands' in self.kwargs:
                bands = self.kwargs['bands']
            elif isinstance(data, SpyFile) and \
                    'default bands' in data.metadata:
                bands = list(map(int, data.metadata['default bands']))
            else:
                bands = list(range(3))
            image = make_pil_image(data, bands)

        # Read each image so it displays properly when viewed from the outside
        # of the cube with corners rendered from lower left CCW to upper left.

        # Read data for sides of cube
        sides = [np.fliplr(np.rot90(data[s[0] - 1, :, :].squeeze(), 3))]   # front face
        sides.append(np.rot90(data[:, s[1] - 1, :].squeeze(), 3))  # right face
        sides.append(np.rot90(data[0, :, :].squeeze(), 3))      # back face
        sides.append(np.fliplr(np.rot90(data[:, 0, :].squeeze(), 3)))      # left face

        # Create images for sides of cube
        scaleMin = min([min(side.ravel()) for side in sides])
        scaleMax = max([max(side.ravel()) for side in sides])
        scale.set_range(scaleMin, scaleMax)
        sideImages = [make_pil_image(side, color_scale=scale, auto_scale=0)
                      for side in sides]
        images = [image] + sideImages

        self.textures = gl.glGenTextures(6)
        texImages = []
        (a, b, c) = data.shape
        texSizes = [(b, a), (b, c), (a, c), (b, c), (a, c), (b, a)]
        for i in range(len(images)):
            try:
                # API change for Pillow
                img = images[i].tobytes("raw", "RGBX", 0, -1)
            except:
                # Fall back to old PIL API
                img = images[i].tostring("raw", "RGBX", 0, -1)
            (dim_x, dim_y) = images[i].size
            texImages.append(img)

            # Create Linear Filtered Texture
            gl.glBindTexture(gl.GL_TEXTURE_2D, int(self.textures[i]))
            gl.glTexParameteri(
                gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
                               gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_R,
                               gl.GL_CLAMP)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S,
                               gl.GL_CLAMP)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T,
                               gl.GL_CLAMP)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, 3, dim_x, dim_y,
                            0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, texImages[i])

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
        """Initialize OpenGL for use in the window."""
        import OpenGL.GL as gl
        import OpenGL.GLU as glu
        self.load_textures()
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glClearColor(*self.clear_color)
        gl.glClearDepth(1.0)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glShadeModel(gl.GL_SMOOTH)

        gl.glMatrixMode(gl.GL_PROJECTION)
        # Reset The projection matrix
        gl.glLoadIdentity()
        # Calculate aspect ratio of the window
        (width, height) = self.canvas.GetClientSize()
        glu.gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (0.5, 0.5, 0.5, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (0.0, 0.0, 2.0, 1.0))
        gl.glEnable(gl.GL_LIGHT0)

    def on_paint(self, event):
        """Process the drawing event."""
        import OpenGL.GL as gl
        import OpenGL.GLU as glu
        self.canvas.SetCurrent(self.canvas.context)

        if not self.gl_initialized:
            self.initgl()
            self.gl_initialized = True
            self.print_help()

        if self.light:
            gl.glEnable(gl.GL_LIGHTING)
        else:
            gl.glDisable(gl.GL_LIGHTING)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()
        gl.glPushMatrix()
        glu.gluLookAt(*(list(rtp_to_xyz(
            *self.camera_pos_rtp)) + list(self.target_pos) + list(self.up)))

        self.draw_cube()

        gl.glPopMatrix()
        gl.glFlush()
        self.SwapBuffers()
        event.Skip()

    def draw_cube(self, *args, **kwargs):
        import OpenGL.GL as gl
        # Determine cube proportions
        divisor = max(self.hsi.shape[:2])
        hw, hh = [float(x) / divisor for x in self.hsi.shape[:2]]
        hz = self.cubeHeight

        # Top Face (note that the texture's corners have to match the quad's)
        gl.glBindTexture(gl.GL_TEXTURE_2D, int(self.textures[0]))
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0.0, 0.0)
        gl.glVertex3f(hw, -hh, hz)  # Bottom Left Of The Texture and Quad
        gl.glTexCoord2f(1.0, 0.0)
        gl.glVertex3f(hw, hh, hz)  # Bottom Right Of The Texture and Quad
        gl.glTexCoord2f(1.0, 1.0)
        gl.glVertex3f(
            -hw, hh, hz)  # Top Right Of The Texture and Quad
        gl.glTexCoord2f(0.0, 1.0)
        gl.glVertex3f(
            -hw, -hh, hz)  # Top Left Of The Texture and Quad
        gl.glEnd()

        # Far Face
        gl.glBindTexture(gl.GL_TEXTURE_2D, int(self.textures[3]))
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0.0, 0.0)
        gl.glVertex3f(
            -hw, hh, -hz)  # Top Left Of The Texture and Quad
        gl.glTexCoord2f(1.0, 0.0)
        gl.glVertex3f(
            -hw, -hh, -hz)  # Bottom Left Of The Texture and Quad
        gl.glTexCoord2f(1.0, 1.0)
        gl.glVertex3f(
            -hw, -hh, hz)  # Bottom Right Of The Texture and Quad
        gl.glTexCoord2f(0.0, 1.0)
        gl.glVertex3f(
            -hw, hh, hz)  # Top Right Of The Texture and Quad
        gl.glEnd()

        # Near Face
        gl.glBindTexture(gl.GL_TEXTURE_2D, int(self.textures[1]))
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0.0, 0.0)
        gl.glVertex3f(
            hw, -hh, -hz)  # Top Right Of The Texture and Quad
        gl.glTexCoord2f(1.0, 0.0)
        gl.glVertex3f(
            hw, hh, -hz)  # Top Left Of The Texture and Quad
        gl.glTexCoord2f(1.0, 1.0)
        gl.glVertex3f(
            hw, hh, hz)  # Bottom Left Of The Texture and Quad
        gl.glTexCoord2f(0.0, 1.0)
        gl.glVertex3f(
            hw, -hh, hz)  # Bottom Right Of The Texture and Quad
        gl.glEnd()

        # Right face
        gl.glBindTexture(gl.GL_TEXTURE_2D, int(self.textures[2]))
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0.0, 0.0)
        gl.glVertex3f(
            hw, hh, -hz)  # Bottom Right Of The Texture and Quad
        gl.glTexCoord2f(1.0, 0.0)
        gl.glVertex3f(
            -hw, hh, -hz)  # Top Right Of The Texture and Quad
        gl.glTexCoord2f(1.0, 1.0)
        gl.glVertex3f(
            -hw, hh, hz)  # Top Left Of The Texture and Quad
        gl.glTexCoord2f(0.0, 1.0)
        gl.glVertex3f(
            hw, hh, hz)  # Bottom Left Of The Texture and Quad
        gl.glEnd()

        # Left Face
        gl.glBindTexture(gl.GL_TEXTURE_2D, int(self.textures[4]))
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0.0, 0.0)
        gl.glVertex3f(
            -hw, -hh, -hz)  # Bottom Left Of The Texture and Quad
        gl.glTexCoord2f(1.0, 0.0)
        gl.glVertex3f(
            hw, -hh, -hz)  # Bottom Right Of The Texture and Quad
        gl.glTexCoord2f(1.0, 1.0)
        gl.glVertex3f(
            hw, -hh, hz)  # Top Right Of The Texture and Quad
        gl.glTexCoord2f(0.0, 1.0)
        gl.glVertex3f(
            -hw, -hh, hz)  # Top Left Of The Texture and Quad
        gl.glEnd()

        # Bottom Face
        gl.glBindTexture(gl.GL_TEXTURE_2D, int(self.textures[0]))
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0.0, 0.0)
        gl.glVertex3f(
            hw, -hh, -hz)  # Bottom Left Of The Texture and Quad
        gl.glTexCoord2f(1.0, 0.0)
        gl.glVertex3f(
            hw, hh, -hz)  # Bottom Right Of The Texture and Quad
        gl.glTexCoord2f(1.0, 1.0)
        gl.glVertex3f(
            -hw, hh, -hz)  # Top Right Of The Texture and Quad
        gl.glTexCoord2f(0.0, 1.0)
        gl.glVertex3f(
            -hw, -hh, -hz)  # Top Left Of The Texture and Quad
        gl.glEnd()

    def on_resize(self, event):
        """Process the resize event."""

        if wx.VERSION >= (2, 9) or self.canvas.GetContext():
            self.canvas.SetCurrent(self.canvas.context)
            self.Show()
            size = self.canvas.GetClientSize()
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
        key = event.GetKeyCode()
        if key == ord('t'):
            self.cubeHeight += 0.1
        elif key == ord('g'):
            self.cubeHeight -= 0.1
        elif key == ord('l'):
            self.light = not self.light
        elif key == ord('h'):
            self.print_help()
#        self.on_draw()
        self.on_paint(event)

        if key == ord('q'):
            self.Destroy()

    def print_help(self):
        print()
        print('Mouse Functions:')
        print('----------------')
        print('left-click & drag        ->   Rotate cube')
        print('CTRL+left-click & drag   ->   Zoom in/out')
        print('SHIFT+left-click & drag  ->  Pan')
        print()
        print('Keybinds:')
        print('---------')
        print('l       -> toggle light')
        print('t/g     -> stretch/compress z-dimension')
        print('h       -> print help message')
        print('q       -> close window')
        print()
