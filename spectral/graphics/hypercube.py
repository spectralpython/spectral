#########################################################################
#
#   hypercube.py - This file is part of the Spectral Python (SPy)
#   package.
#
#   Copyright (C) 2001-2012 Thomas Boggs
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
#
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
# This code was created by Richard Campbell '99 (ported to Python/PyOpenGL by John Ferguson 2000)
#
# The port was based on the lesson5 tutorial module by Tony Colston (tonetheman@hotmail.com).  
#
# If you've found this code useful, please let me know (email John Ferguson at hakuin@voicenet.com).
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

'''
Code for rendering and manipulating hypercubes. Most users will only need to
call the function "hypercube".
'''

try:
    import wx
    from wx import glcanvas
except ImportError:
    raise ImportError, "Required dependency wx.glcanvas not present"

DEFAULT_WIN_SIZE = (500, 500)		# Default dimensions of image frame
DEFAULT_TEXTURE_SIZE = (256, 256)	# Default size of textures on cube faces

def rtp_to_xyz(r, theta, phi):
    '''Convert spherical polar coordinates to Cartesian'''
    from math import pi, cos, sin
    theta *= pi / 180.0
    phi *= pi / 180.0
    s = r * sin(theta)
    return [s * cos(phi), s * sin(phi), r * cos(theta)]

def xyz_to_rtp(x, y, z):
    '''Convert Cartesian coordinates to Spherical Polar.'''
    from math import asin, acos, sqrt, pi
    r = sqrt(x * x + y * y + z * z)
    rho = sqrt(x * x + y * y)
    phi = asin(y / rho) * 180. / pi
    if x < 0.0:
	phi += 180 
    theta = acos(z / r) * 180. / pi
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
	import numpy as np
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
		# Mouse movement pans target position in the plane of the window
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

class WxHypercubeFrame(wx.Frame):
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
        super(WxHypercubeFrame, self).__init__(parent, id, self.title,
					       wx.DefaultPosition,
					       wx.Size(*self.size),
					       style,
					       kwargs.get('name', 'Hypercube'))
        
        self.gl_initialized = False
        attribs = (glcanvas.WX_GL_RGBA, # RGBA
                   glcanvas.WX_GL_DOUBLEBUFFER, # Double Buffered
                   glcanvas.WX_GL_DEPTH_SIZE, 32) # 32 bit
        self.canvas = glcanvas.GLCanvas(self, attribList=attribs)

	# These members can be modified before calling the show method.
	self.clear_color = (0., 0., 0., 1.)
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
        from Image import open
	import OpenGL.GL as gl
        import spectral
	import graphics
        from spectral.graphics.colorscale import default_color_scale


        global DEFAULT_TEXTURE_SIZE

        if self.kwargs.has_key('textureSize'):
            (dimX, dimY) = self.kwargs['textureSize']
        else:
            (dimX, dimY) = DEFAULT_TEXTURE_SIZE

        if self.kwargs.has_key('scale'):
            scale = self.kwargs['scale']
        else:
            scale = default_color_scale

        data = self.hsi
        s = data.shape

        # Create image for top of cube
        if self.kwargs.has_key('top'):
            image = self.kwargs['top']
        else:
            if self.kwargs.has_key('bands'):
                bands = self.kwargs['bands']
            elif isinstance(data, spectral.io.SpyFile) and data.metadata.has_key('default bands'):
                bands = map(int, data.metadata['default bands'])
            else:
                bands = range(3)
            image = graphics.make_pil_image(data, bands, format='bmp')

        # Read data for sides of cube
        sides = [data[s[0] - 1, :, :].squeeze()]		# front face
        sides.append(data[:, s[1] - 1, :].squeeze())		# right face
        sides.append(data[0, :, :].squeeze())			# back face
        sides.append(data[:, 0, :].squeeze())			# left face

        # Create images for sides of cube
        scaleMin = min([min(side.ravel()) for side in sides])
        scaleMax = max([max(side.ravel()) for side in sides])
        scale = default_color_scale
        scale.set_range(scaleMin, scaleMax)
        sideImages = [graphics.make_pil_image(side, colorScale=scale, autoScale=1, format='bmp') for side in sides]
        images = [image] + sideImages + [image]

        self.textures = gl.glGenTextures(6)
        texImages = []
        (a, b, c) = data.shape
        texSizes = [(b, a), (b, c), (a, c), (b, c), (a, c), (b, a)]
        for i in range(len(images)):
            img = images[i].resize((dimX, dimY))
            img = img.tostring("raw", "RGBX", 0, -1)
            texImages.append(img)
            
            # Create Linear Filtered Texture 
            gl.glBindTexture(gl.GL_TEXTURE_2D, long(self.textures[i]))
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, 3, dimX, dimY, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, texImages[i])

    def GetGLExtents(self):
        """Get the extents of the OpenGL canvas."""
        return 
    
    def SwapBuffers(self):
        """Swap the OpenGL buffers."""
        self.canvas.SwapBuffers()
    
    def on_erase_background(self, event):
        """Process the erase background event."""
        pass # Do nothing, to avoid flashing on MSWin
    
    def initgl(self):
        """Initialize OpenGL for use in the window."""
	import OpenGL.GL as gl
	import OpenGL.GLU as glu
        self.load_textures()
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)	# This Will Clear The Background Color To Black
        gl.glClearDepth(1.0)			# Enables Clearing Of The Depth Buffer
        gl.glDepthFunc(gl.GL_LESS)			# The Type Of Depth Test To Do
        gl.glEnable(gl.GL_DEPTH_TEST)			# Enables Depth Testing
        gl.glShadeModel(gl.GL_SMOOTH)			# Enables Smooth Color Shading
            
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()			# Reset The Projection Matrix
        # Calculate The Aspect Ratio Of The Window
        (width, height) = self.canvas.GetClientSize()
        glu.gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)

        gl.glMatrixMode(gl.GL_MODELVIEW)

        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (0.5, 0.5, 0.5, 1.0))	# Setup The Ambient Light 
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))	# Setup The Diffuse Light 
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (0.0, 0.0, 2.0, 1.0))	# Position The Light 
        gl.glEnable(gl.GL_LIGHT0)					# Enable Light One 

    def on_paint(self, event):
        """Process the drawing event."""
	import OpenGL.GL as gl
	import OpenGL.GLU as glu
        self.canvas.SetCurrent()
        
        if not self.gl_initialized:
            self.initgl()
            self.gl_initialized = True
            self.print_help()
        
        if self.light:
            gl.glEnable(gl.GL_LIGHTING)
        else:
            gl.glDisable(gl.GL_LIGHTING)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)	# Clear The Screen And The Depth Buffer
        gl.glLoadIdentity()					# Reset The View

	gl.glPushMatrix()
	glu.gluLookAt(*(list(rtp_to_xyz(*self.camera_pos_rtp)) + list(self.target_pos) + list(self.up)))

        self.draw_cube()

	gl.glPopMatrix()
	gl.glFlush()
	self.SwapBuffers()
	event.Skip()
    
    def draw_cube(self, *args, **kwargs):
	import OpenGL.GL as gl
        # Determine cube proportions
        divisor = max(self.hsi.shape[:2])
        hh, hw = [float(x) / divisor for x in self.hsi.shape[:2]]
        hz = self.cubeHeight

        # Top Face (note that the texture's corners have to match the quad's corners)
        gl.glBindTexture(gl.GL_TEXTURE_2D, long(self.textures[0]))
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0.0, 0.0); gl.glVertex3f(-hw, -hh,  hz)	# Bottom Left Of The Texture and Quad
        gl.glTexCoord2f(1.0, 0.0); gl.glVertex3f( hw, -hh,  hz)	# Bottom Right Of The Texture and Quad
        gl.glTexCoord2f(1.0, 1.0); gl.glVertex3f( hw,  hh,  hz)	# Top Right Of The Texture and Quad
        gl.glTexCoord2f(0.0, 1.0); gl.glVertex3f(-hw,  hh,  hz)	# Top Left Of The Texture and Quad
        gl.glEnd();


        # Far Face
        gl.glBindTexture(gl.GL_TEXTURE_2D, long(self.textures[3]))
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(1.0, 1.0); gl.glVertex3f(-hw,  hh, -hz)	# Top Left Of The Texture and Quad
        gl.glTexCoord2f(0.0, 1.0); gl.glVertex3f(-hw,  hh,  hz)	# Bottom Left Of The Texture and Quad
        gl.glTexCoord2f(0.0, 0.0); gl.glVertex3f( hw,  hh,  hz)	# Bottom Right Of The Texture and Quad
        gl.glTexCoord2f(1.0, 0.0); gl.glVertex3f( hw,  hh, -hz)	# Top Right Of The Texture and Quad
        gl.glEnd();

        # Near Face       
        gl.glBindTexture(gl.GL_TEXTURE_2D, long(self.textures[1]))
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(1.0, 1.0); gl.glVertex3f(-hw, -hh, -hz)	# Top Right Of The Texture and Quad
        gl.glTexCoord2f(1.0, 0.0); gl.glVertex3f( hw, -hh, -hz)	# Top Left Of The Texture and Quad
        gl.glTexCoord2f(0.0, 0.0); gl.glVertex3f( hw, -hh,  hz)	# Bottom Left Of The Texture and Quad
        gl.glTexCoord2f(0.0, 1.0); gl.glVertex3f(-hw, -hh,  hz)	# Bottom Right Of The Texture and Quad
        gl.glEnd();

        # Right face
        gl.glBindTexture(gl.GL_TEXTURE_2D, long(self.textures[2]))
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(1.0, 0.0); gl.glVertex3f( hw, -hh, -hz)	# Bottom Right Of The Texture and Quad
        gl.glTexCoord2f(1.0, 1.0); gl.glVertex3f( hw,  hh, -hz)	# Top Right Of The Texture and Quad
        gl.glTexCoord2f(0.0, 1.0); gl.glVertex3f( hw,  hh,  hz)	# Top Left Of The Texture and Quad
        gl.glTexCoord2f(0.0, 0.0); gl.glVertex3f( hw, -hh,  hz)	# Bottom Left Of The Texture and Quad
        gl.glEnd();

        # Left Face
        gl.glBindTexture(gl.GL_TEXTURE_2D, long(self.textures[4]))
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(1.0, 0.0); gl.glVertex3f(-hw, -hh, -hz)	# Bottom Left Of The Texture and Quad
        gl.glTexCoord2f(0.0, 0.0); gl.glVertex3f(-hw, -hh,  hz)	# Bottom Right Of The Texture and Quad
        gl.glTexCoord2f(0.0, 1.0); gl.glVertex3f(-hw,  hh,  hz)	# Top Right Of The Texture and Quad
        gl.glTexCoord2f(1.0, 1.0); gl.glVertex3f(-hw,  hh, -hz)	# Top Left Of The Texture and Quad
        gl.glEnd();

    def on_resize(self, event):
        """Process the resize event."""
        if self.canvas.GetContext():
            # Make sure the frame is shown before calling SetCurrent.
            self.Show()
            self.canvas.SetCurrent()
            size = self.canvas.GetClientSize()
            self.resize(size.width, size.height)
            self.canvas.Refresh(False)
        event.Skip()
    
    def resize(self, width, height):
        """Reshape the OpenGL viewport based on the dimensions of the window."""
	import OpenGL.GL as gl
        self.size = (width, height)
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
	glu.gluPerspective(self.fovy, float(width) / height,
			   self.znear, self.zfar)
        
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
    
    def on_char(self,event):
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
        print
	print 'Mouse Functions:'
	print '----------------'
	print 'left-click & drag        ->   Rotate cube'
	print 'CTRL+left-click & drag   ->   Zoom in/out'
	print 'SHIFT+left-click & drag  ->  Pan'
	print
        print 'Keybinds:'
        print '---------'
        print 'l       -> toggle light'
        print 't/g     -> stretch/compress z-dimension'
        print 'h       -> print help message'
        print 'q       -> close window'
        print

class HypercubeFunctor:
    '''A functor used to create the new window in the second thread.'''
    def __init__(self, data, *args, **kwargs):
        self.data = data
        self.args = args
        self.kwargs = kwargs
    def __call__(self):
        frame = WxHypercubeFrame(self.data, None, -1, *self.args, **self.kwargs)
        return frame

def hypercube(data, *args, **kwargs):
    '''Renders an interactive 3D hypercube in a new window.

    Arguments:

	`data` (:class:`spectral.Image` or :class:`numpy.ndarray`):
	
	    Source image data to display.  `data` can be and instance of a
	    :class:`spectral.Image` (e.g., :class:`spectral.SpyFile` or
	    :class:`spectral.ImageArray`) or a :class:`numpy.ndarray`. `source`
	    must have shape `MxN` or `MxNxB`.

    Keyword Arguments:

        `bands` (3-tuple of ints):
	
	    3-tuple specifying which bands from the image data should be
	    displayed on top of the cube.

        `top` (:class:`PIL.Image`):
	
	    An alternate bitmap to display on top of the cube.

        `scale` (:class:`spectral.ColorScale`)
	
	    A color scale to be used for color in the sides of the cube. If this
	    keyword is not specified, :obj:`spectral.graphics.colorscale.defaultColorScale`
	    is used.
	
	`size` (2-tuple of ints):
	
	    Width and height (in pixels) for initial size of the new window.

        `title` (str):
	
	    Title text to display in the new window frame.
    
    This function opens a new window, renders a 3D hypercube, and accepts
    keyboard input to manipulate the view of the hypercube.  Accepted keyboard
    inputs are printed to the console output.  Focus must be on the 3D window
    to accept keyboard input.  To avoid unecessary :mod:`PyOpenGl` dependency,
    `hypercube` is not imported into the main `spectral` namespace by default so
    you must import it::
    
	from spectral.graphics.hypercube import hypercube
    '''
    import spectral
    import time
    from spectral.graphics import spywxpython

    # Initialize the display thread if it isn't already
    if spywxpython.viewer == None:
	spectral.init_graphics()
	time.sleep(3)

    functor = HypercubeFunctor(data, *args, **kwargs)
    spywxpython.viewer.view(None, function=functor)
