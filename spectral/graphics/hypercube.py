'''
Code for rendering and manipulating hypercubes.
Most users will only need to call the function "hypercube".
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import math
import numpy as np

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtWidgets import QApplication
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
except ImportError:
    raise ImportError("Required dependency PySide6 not present")

from .. import settings
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


def ensure_qt_event_loop():
    '''Ensures a QApplication exists and that its event loop is being
    pumped.

    When running under IPython, this enables IPython's Qt GUI integration
    (equivalent to `%gui qt`) so that Qt events (repaints, mouse clicks, key
    presses, ...) are processed between input prompts instead of only being
    flushed when the interpreter exits.
    '''
    if QApplication.instance() is None:
        QApplication([])

    try:
        from IPython import get_ipython
        ip = get_ipython()
    except ImportError:
        ip = None

    if ip is not None:
        ip.enable_gui('qt')

    return QApplication.instance()


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
        pos = event.position()
        self.event_position = (pos.x(), pos.y())
        self.position = (pos.x(), pos.y())
        self.left = DOWN

    def left_up(self, event):
        pos = event.position()
        self.position = (pos.x(), pos.y())
        self.left = UP

    def motion(self, event):
        '''Handles panning & zooming for mouse click+drag events.'''
        if DOWN not in (self.left, self.right):
            return
        (w, h) = self.window.win_size
        pos = event.position()
        x, y = pos.x(), pos.y()
        dx = x - self.position[0]
        dy = y - self.position[1]
        modifiers = event.modifiers()
        if self.left == DOWN:
            if modifiers & Qt.ControlModifier:
                # Mouse movement zooms in/out relative to target position
                if dx != 0.0:
                    self.window.camera_pos_rtp[0] *= (float(w - dx) / w)
            elif modifiers & Qt.ShiftModifier:
                # Mouse movement pans target position in  plane of the window
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
        self.position = (x, y)
        self.window.update()


class HypercubeWindow(QOpenGLWidget, SpyWindow):
    """A simple class for using OpenGL with PySide6."""

    def __init__(self, data, parent, id, *args, **kwargs):
        global DEFAULT_WIN_SIZE

        self._app = ensure_qt_event_loop()

        self.kwargs = kwargs
        self.win_size = kwargs.get('size', DEFAULT_WIN_SIZE)
        self.title = kwargs.get('title', 'Hypercube')

        super().__init__(parent)

        self.setWindowTitle(self.title)
        self.resize(*self.win_size)

        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(settings.GL_DEPTH_SIZE)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
        self.setFormat(fmt)

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

        self.setFocusPolicy(Qt.StrongFocus)

    def Show(self, show=True):
        """Show (or hide) the window."""
        if show:
            self.show()
            self.setFocus()
        else:
            self.hide()

    def Raise(self):
        """Raise the window to the top of the window stack."""
        self.raise_()
        self.activateWindow()

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
        for i in range(len(images)):
            img = images[i].tobytes("raw", "RGBX", 0, -1)
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

    def initializeGL(self):
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
        width = max(self.width(), 1)
        height = max(self.height(), 1)
        glu.gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (0.5, 0.5, 0.5, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (0.0, 0.0, 2.0, 1.0))
        gl.glEnable(gl.GL_LIGHT0)

        self.print_help()

    def paintGL(self):
        """Process the drawing event."""
        import OpenGL.GL as gl
        import OpenGL.GLU as glu

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

    def resizeGL(self, width, height):
        """Reshape the OpenGL viewport based on dimensions of the window."""
        import OpenGL.GL as gl
        import OpenGL.GLU as glu
        self.win_size = (width, height)
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(self.fovy, float(width) / max(height, 1),
                           self.znear, self.zfar)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() == Qt.LeftButton:
            self.mouse_handler.left_down(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_handler.left_up(event)

    def mouseMoveEvent(self, event):
        self.mouse_handler.motion(event)

    def keyPressEvent(self, event):
        key = event.text()
        if key == 't':
            self.cubeHeight += 0.1
        elif key == 'g':
            self.cubeHeight -= 0.1
        elif key == 'l':
            self.light = not self.light
        elif key == 'h':
            self.print_help()
        elif key == 'q':
            self.close()
            return
        self.update()

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
