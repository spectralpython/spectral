#########################################################################
#
#   Hypercube.py - This file is part of the Spectral Python (SPy)
#   package.
#
#   Copyright (C) 2001-2008 Thomas Boggs
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

try:
    from OpenGL.GL import *
    from OpenGL.GLUT import *
    from OpenGL.GLU import *
except ImportError:
    raise ImportError, "Required dependency OpenGL not present"

DEFAULT_WIN_SIZE = (500, 500)                   # Default dimensions of image frame
DEFAULT_TEXTURE_SIZE = (256, 256)               # Default size of textures on cube faces


class WxHypercubeFrame(wx.Frame):
    """A simple class for using OpenGL with wxPython."""
    
    def __init__(self, data, parent, id, *args, **kwargs):
        global DEFAULT_WIN_SIZE

        self.kwargs = kwargs
        if kwargs.has_key('size'):
            size = wx.Size(*kwargs['size'])
        else:
            size = wx.Size(*DEFAULT_WIN_SIZE)
        if kwargs.has_key('title'):
            title = kwargs['title']
        else:
            title = 'Hypercube'

        #
        # Forcing a specific style on the window.
        #   Should this include styles passed?
        style = wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE
        
        super(WxHypercubeFrame, self).__init__(parent, id, title, wx.DefaultPosition, size, wx.DEFAULT_FRAME_STYLE, kwargs.get('name', 'Hypercube'))
        
        self.GLinitialized = False
        attribList = (glcanvas.WX_GL_RGBA, # RGBA
                      glcanvas.WX_GL_DOUBLEBUFFER, # Double Buffered
                      glcanvas.WX_GL_DEPTH_SIZE, 32) # 32 bit
        
        #
        # Create the canvas
        self.canvas = glcanvas.GLCanvas(self, attribList=attribList)

        self.hsi = data
        self.cubeHeight = 1.0
        self.rotation = [-60, 0, -30]
        self.distance = -5
        self.light = False

        self.texturesLoaded = False
        
        #
        # Set the event handlers.
        self.canvas.Bind(wx.EVT_ERASE_BACKGROUND, self.processEraseBackgroundEvent)
        self.canvas.Bind(wx.EVT_SIZE, self.processSizeEvent)
        self.canvas.Bind(wx.EVT_PAINT, self.onPaint)
        self.canvas.Bind(wx.EVT_CHAR, self.onChar)

    def LoadTextures(self):
        #global texture
        from Image import open
        import Spectral
        from Spectral.Graphics.ColorScale import defaultColorScale

        global DEFAULT_TEXTURE_SIZE

        if self.kwargs.has_key('textureSize'):
            (dimX, dimY) = self.kwargs['textureSize']
        else:
            (dimX, dimY) = DEFAULT_TEXTURE_SIZE

        if self.kwargs.has_key('scale'):
            scale = self.kwargs['scale']
        else:
            scale = defaultColorScale

        data = self.hsi
        s = data.shape

        # Create image for top of cube
        if self.kwargs.has_key('top'):
            image = self.kwargs['top']
        else:
            if self.kwargs.has_key('bands'):
                bands = self.kwargs['bands']
            elif isinstance(data, Spectral.Io.SpyFile) and data.metadata.has_key('default bands'):
                bands = map(int, data.metadata['default bands'])
            else:
                bands = range(3)
            image = Spectral.makePilImage(data, bands, format='bmp')

        # Read data for sides of cube
        sides = [data[s[0] - 1, :, :].squeeze()]		# front face
        sides.append(data[:, s[1] - 1, :].squeeze())		# right face
        sides.append(data[0, :, :].squeeze())			# back face
        sides.append(data[:, 0, :].squeeze())			# left face

        # Create images for sides of cube
        scaleMin = min([min(side.ravel()) for side in sides])
        scaleMax = max([max(side.ravel()) for side in sides])
        scale = defaultColorScale
        scale.setRange(scaleMin, scaleMax)
        sideImages = [Spectral.makePilImage(side, colorScale=scale, autoScale=1, format='bmp') for side in sides]
        images = [image] + sideImages + [image]

        self.textures = glGenTextures(6)
        texImages = []
        (a, b, c) = data.shape
        texSizes = [(b, a), (b, c), (a, c), (b, c), (a, c), (b, a)]
        for i in range(len(images)):
            img = images[i].resize((dimX, dimY))
            img = img.tostring("raw", "RGBX", 0, -1)
            texImages.append(img)
            
            # Create Linear Filtered Texture 
            glBindTexture(GL_TEXTURE_2D, long(self.textures[i]))
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, 3, dimX, dimY, 0, GL_RGBA, GL_UNSIGNED_BYTE, texImages[i])
            
##        # Create Texture	
##        glBindTexture(GL_TEXTURE_2D, glGenTextures(1))   # 2d texture (x and y size)
##            
##        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
##        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
##        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
##        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
##        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
##        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
##        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
##        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
##        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

    def GetGLExtents(self):
        """Get the extents of the OpenGL canvas."""
        return 
    
    def SwapBuffers(self):
        """Swap the OpenGL buffers."""
        self.canvas.SwapBuffers()
    
    #
    # wxPython Window Handlers
    
    def processEraseBackgroundEvent(self, event):
        """Process the erase background event."""
        pass # Do nothing, to avoid flashing on MSWin
    
    def processSizeEvent(self, event):
        """Process the resize event."""
        if self.canvas.GetContext():
            # Make sure the frame is shown before calling SetCurrent.
            self.Show()
            self.canvas.SetCurrent()

            size = self.canvas.GetClientSize()
            self.OnReshape(size.width, size.height)
            self.canvas.Refresh(False)
        event.Skip()
    
    def onPaint(self, event):
        """Process the drawing event."""
        self.canvas.SetCurrent()
        
        # This is a 'perfect' time to initialize OpenGL ... only if we need to
        if not self.GLinitialized:
            self.OnInitGL()
            self.GLinitialized = True
            self.printHelp()
        
        if self.light:
            glEnable(GL_LIGHTING)
        else:
            glDisable(GL_LIGHTING)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)	# Clear The Screen And The Depth Buffer
        glLoadIdentity()					# Reset The View
        self.OnDraw()

        event.Skip()
    
    #
    # GLFrame OpenGL Event Handlers
    
    def OnInitGL(self):
        """Initialize OpenGL for use in the window."""
        self.LoadTextures()
        glEnable(GL_TEXTURE_2D)
        glClearColor(0.0, 0.0, 0.0, 0.0)	# This Will Clear The Background Color To Black
        glClearDepth(1.0)			# Enables Clearing Of The Depth Buffer
        glDepthFunc(GL_LESS)			# The Type Of Depth Test To Do
        glEnable(GL_DEPTH_TEST)			# Enables Depth Testing
        glShadeModel(GL_SMOOTH)			# Enables Smooth Color Shading
            
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()			# Reset The Projection Matrix
        # Calculate The Aspect Ratio Of The Window
        (width, height) = self.canvas.GetClientSize()
        gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)

        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.5, 0.5, 0.5, 1.0))	# Setup The Ambient Light 
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))	# Setup The Diffuse Light 
        glLightfv(GL_LIGHT0, GL_POSITION, (0.0, 0.0, 2.0, 1.0))	# Position The Light 
        glEnable(GL_LIGHT0)					# Enable Light One 

    
    def OnReshape(self, width, height):
        """Reshape the OpenGL viewport based on the dimensions of the window."""
        glViewport(0, 0, width, height)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
    
    def OnDraw(self, *args, **kwargs):

        # Determine cube proportions
        divisor = max(self.hsi.shape[:2])
        hh, hw = [float(x) / divisor for x in self.hsi.shape[:2]]
        hz = self.cubeHeight

        # Cube orientation
        glTranslatef(0.0, 0.0, self.distance)			# Move Into The Screen
        glRotatef(self.rotation[0],1.0,0.0,0.0)			# Rotate The Cube On It's X Axis
        glRotatef(self.rotation[1],0.0,1.0,0.0)			# Rotate The Cube On It's Y Axis
        glRotatef(self.rotation[2],0.0,0.0,1.0)			# Rotate The Cube On It's Z Axis

        # Top Face (note that the texture's corners have to match the quad's corners)
        glBindTexture(GL_TEXTURE_2D, long(self.textures[0]))
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex3f(-hw, -hh,  hz)	# Bottom Left Of The Texture and Quad
        glTexCoord2f(1.0, 0.0); glVertex3f( hw, -hh,  hz)	# Bottom Right Of The Texture and Quad
        glTexCoord2f(1.0, 1.0); glVertex3f( hw,  hh,  hz)	# Top Right Of The Texture and Quad
        glTexCoord2f(0.0, 1.0); glVertex3f(-hw,  hh,  hz)	# Top Left Of The Texture and Quad
        glEnd();

        # Bottom Face
        glBindTexture(GL_TEXTURE_2D, long(self.textures[5]))
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex3f(-hw, -hh, -hz)	# Bottom Left Of The Texture and Quad
        glTexCoord2f(1.0, 0.0); glVertex3f( hw, -hh, -hz)	# Bottom Right Of The Texture and Quad
        glTexCoord2f(1.0, 1.0); glVertex3f( hw,  hh, -hz)	# Top Right Of The Texture and Quad
        glTexCoord2f(0.0, 1.0); glVertex3f(-hw,  hh, -hz)	# Top Left Of The Texture and Quad
        glEnd();

        # Far Face
        glBindTexture(GL_TEXTURE_2D, long(self.textures[3]))
        glBegin(GL_QUADS)
        glTexCoord2f(1.0, 1.0); glVertex3f(-hw,  hh, -hz)	# Top Left Of The Texture and Quad
        glTexCoord2f(0.0, 1.0); glVertex3f(-hw,  hh,  hz)	# Bottom Left Of The Texture and Quad
        glTexCoord2f(0.0, 0.0); glVertex3f( hw,  hh,  hz)	# Bottom Right Of The Texture and Quad
        glTexCoord2f(1.0, 0.0); glVertex3f( hw,  hh, -hz)	# Top Right Of The Texture and Quad
        glEnd();

        # Near Face       
        glBindTexture(GL_TEXTURE_2D, long(self.textures[1]))
        glBegin(GL_QUADS)
        glTexCoord2f(1.0, 1.0); glVertex3f(-hw, -hh, -hz)	# Top Right Of The Texture and Quad
        glTexCoord2f(1.0, 0.0); glVertex3f( hw, -hh, -hz)	# Top Left Of The Texture and Quad
        glTexCoord2f(0.0, 0.0); glVertex3f( hw, -hh,  hz)	# Bottom Left Of The Texture and Quad
        glTexCoord2f(0.0, 1.0); glVertex3f(-hw, -hh,  hz)	# Bottom Right Of The Texture and Quad
        glEnd();

        # Right face
        glBindTexture(GL_TEXTURE_2D, long(self.textures[2]))
        glBegin(GL_QUADS)
        glTexCoord2f(1.0, 0.0); glVertex3f( hw, -hh, -hz)	# Bottom Right Of The Texture and Quad
        glTexCoord2f(1.0, 1.0); glVertex3f( hw,  hh, -hz)	# Top Right Of The Texture and Quad
        glTexCoord2f(0.0, 1.0); glVertex3f( hw,  hh,  hz)	# Top Left Of The Texture and Quad
        glTexCoord2f(0.0, 0.0); glVertex3f( hw, -hh,  hz)	# Bottom Left Of The Texture and Quad
        glEnd();

        # Left Face
        glBindTexture(GL_TEXTURE_2D, long(self.textures[4]))
        glBegin(GL_QUADS)
        glTexCoord2f(1.0, 0.0); glVertex3f(-hw, -hh, -hz)	# Bottom Left Of The Texture and Quad
        glTexCoord2f(0.0, 0.0); glVertex3f(-hw, -hh,  hz)	# Bottom Right Of The Texture and Quad
        glTexCoord2f(0.0, 1.0); glVertex3f(-hw,  hh,  hz)	# Top Right Of The Texture and Quad
        glTexCoord2f(1.0, 1.0); glVertex3f(-hw,  hh, -hz)	# Top Left Of The Texture and Quad
        glEnd();

        self.SwapBuffers()

    def onChar(self,event):
        key = event.GetKeyCode()
        if key == ord('w'):
            self.rotation[0] -= 1
        elif key == ord('s'):
            self.rotation[0] += 1
        elif key == ord('a'):
            self.rotation[2] -= 1
        elif key == ord('d'):
            self.rotation[2] += 1
        elif key == ord('r'):
            self.distance -= 0.1
        elif key == ord('f'):
            self.distance += 0.1
        elif key == ord('t'):
            self.cubeHeight += 0.1
        elif key == ord('g'):
            self.cubeHeight -= 0.1
        elif key == ord('l'):
            self.light = not self.light
        elif key == ord('h'):
            self.printHelp()
        self.OnDraw()
        self.onPaint(event)

        if key == ord('q'):
            self.Destroy()

    def printHelp(self):
        print
        print 'Keybinds:'
        print '---------'
        print 'w/s/a/d -> rotate up/down/left/right'
        print 'r/f     -> increase/decrease distance'
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
    '''
    Renders an interactive hypercube in a new window.

    USAGE: hypercube(data [, **kwargs])

    ARGUMENTS:

        data                A SpyFile object or rank 3 Numeric array

    OPTIONAL KEYWORD ARGUMENTS:

        bands               3-tuple specifying which bands from the image
                            data should be displayed on top of the cube.
        top                 An alternate bitmap to display on the top of
                            the cube.
        scale               A color scale to be used for color in the
                            sides of the cube.
        title               Title text to display in window frame.
    '''
    
    from Spectral.Graphics.SpyWxPython import viewer
    cubeFunctor = HypercubeFunctor(data, *args, **kwargs)
    viewer.view(None, function = cubeFunctor)
