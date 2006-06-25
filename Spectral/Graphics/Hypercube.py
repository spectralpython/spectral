#########################################################################
#
#   Hypercube.py - This file is part of the Spectral Python (SPy)
#   package.
#
#   Copyright (C) 2001-2006 Thomas Boggs
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
''' Code for rendering and manipulating hypercubes.'''

from wxPython.wx import *
from wxPython.glcanvas import *

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

class WxHypercubeWindow(wxGLCanvas):
    ''' A wxWindow that displays an interactive hypercube.'''
    def __init__(self, hsi, parent, *args, **kwargs):
        apply(wxGLCanvas.__init__,(self, parent)+args)
        self.parent = parent

        self.kwargs = kwargs
        self.hsi = hsi
        self.cubeHeight = 1.0
        self.rotation = [-60, 0, -30]
        self.distance = -5
        self.light = False
        
        EVT_SIZE(self,self.wxSize)
        EVT_PAINT(self,self.wxPaint)
        EVT_ERASE_BACKGROUND(self, self.wxEraseBackground)
        EVT_CHAR(self,self.OnChar)

        self.w, self.h = self.GetClientSizeTuple()

        print 'Processing hypercube...',
        self.InitGL()
        print 'done.'

        print 'Press "h" for keybind help'

    def __del__(self):
        self.FinishGL()

    def InitGL(self):				# We call this right after our OpenGL window is created.
        
        self.LoadTextures()

        glEnable(GL_TEXTURE_2D)
        glClearColor(0.0, 0.0, 0.0, 0.0)	# This Will Clear The Background Color To Black
        glClearDepth(1.0)					# Enables Clearing Of The Depth Buffer
        glDepthFunc(GL_LESS)				# The Type Of Depth Test To Do
        glEnable(GL_DEPTH_TEST)				# Enables Depth Testing
        glShadeModel(GL_SMOOTH)				# Enables Smooth Color Shading
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()					# Reset The Projection Matrix
                                                                                # Calculate The Aspect Ratio Of The Window
        gluPerspective(45.0, float(self.w)/float(self.h), 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)

        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.5, 0.5, 0.5, 1.0))		# Setup The Ambient Light 
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))		# Setup The Diffuse Light 
        glLightfv(GL_LIGHT0, GL_POSITION, (0.0, 0.0, 2.0, 1.0))	# Position The Light 
        glEnable(GL_LIGHT0)					# Enable Light One 

    def FinishGL(self):
        """OpenGL closing routine (to be overridden).

        This routine should be overridden if necessary by any OpenGL commands need to be specified when deleting the GLWindow (e.g. deleting Display Lists)."""
        pass

    def LoadTextures(self):
        import Spectral
        import Image
        from ColorScale import defaultColorScale

        # Create raster image for top of cube
        data = self.hsi
        s = data.shape
        images = []
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
        images.append(image)

        if self.kwargs.has_key('scale'):
            scale = self.kwargs['scale']
        else:
            scale = defaultColorScale

        # Now the sides of the cube:

        # TO DO:  flip data.
        tmp = data[s[0] - 1, :, :]			# front face
        Spectral.saveImage('front.bmp', tmp[0,:,:], 'bmp', colorScale=scale, autoScale=1)
        images.append(Spectral.makePilImage(tmp[0, :, :], colorScale=scale, autScale=1, format='bmp'))
        tmp = data[:, s[1] - 1, :]			# right face
        images.append(Spectral.makePilImage(tmp[:, 0, :], colorScale=scale, autScale=1, format='bmp'))
        tmp = data[0, :, :]				# back face
        images.append(Spectral.makePilImage(tmp[0, :, :], colorScale=scale, autScale=1, format='bmp'))
        tmp = data[:, 0, :]				# left face
        images.append(Spectral.makePilImage(tmp[:, 0, :], colorScale=scale, autScale=1, format='bmp'))
        tmp = 0
        images.append(image)				# bottom
        
        self.textures = glGenTextures(6)

        texImages = []
        (a, b, c) = data.shape
        texSizes = [(b, a), (b, c), (a, c), (b, c), (a, c), (b, a)]
        (DIMX, DIMY) = (256, 256)
        for i in range(len(images)):
##            (DIMX, DIMY) = texSizes[i]
            img = images[i].resize((DIMX, DIMY))
            img = img.tostring("raw", "RGBX", 0, -1)
            texImages.append(img)
            
            # Create Linear Filtered Texture 
            glBindTexture(GL_TEXTURE_2D, self.textures[i])
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, 3, DIMX, DIMY, 0, GL_RGBA, GL_UNSIGNED_BYTE, texImages[i])

    def DrawGL(self):
        """OpenGL drawing routine (to be overridden).
        This routine, containing purely OpenGL commands, should be overridden by the user to draw the GL scene. If it is not overridden, it defaults to drawing a colour cube."""


        self.SetCurrent()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)	# Clear The Screen And The Depth Buffer
        glLoadIdentity()					# Reset The View
        glTranslatef(0.0,0.0,self.distance)			# Move Into The Screen
        glRotatef(self.rotation[0],1.0,0.0,0.0)			# Rotate The Cube On It's X Axis
        glRotatef(self.rotation[1],0.0,1.0,0.0)			# Rotate The Cube On It's Y Axis
        glRotatef(self.rotation[2],0.0,0.0,1.0)			# Rotate The Cube On It's Z Axis
        
        if self.light:
            glEnable(GL_LIGHTING)
        else:
            glDisable(GL_LIGHTING)

        self.DrawCube()

        #  since this is double buffered, swap the buffers to display what just got drawn. 
        self.SwapBuffers()

    def DrawCube(self):

        divisor = max(self.hsi.shape[:2])
        hh, hw = [float(x) / divisor for x in self.hsi.shape[:2]]
        hz = self.cubeHeight
        
        # Front Face (note that the texture's corners have to match the quad's corners)
        glBindTexture(GL_TEXTURE_2D, self.textures[0])
        glBegin(GL_QUADS)				# Start Drawing The Cube
        glTexCoord2f(0.0, 0.0); glVertex3f(-hw, -hh,  hz)	# Bottom Left Of The Texture and Quad
        glTexCoord2f(1.0, 0.0); glVertex3f( hw, -hh,  hz)	# Bottom Right Of The Texture and Quad
        glTexCoord2f(1.0, 1.0); glVertex3f( hw,  hh,  hz)	# Top Right Of The Texture and Quad
        glTexCoord2f(0.0, 1.0); glVertex3f(-hw,  hh,  hz)	# Top Left Of The Texture and Quad
        glEnd();						# Done Drawing The face
        
        # Back Face
        ##	glBindTexture(GL_TEXTURE_2D, self.textures[0])
        ##	glBegin(GL_QUADS)				# Start Drawing The Cube
        ##	glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, -1.0)	# Bottom Right Of The Texture and Quad
        ##	glTexCoord2f(0.0, 1.0); glVertex3f(-1.0,  1.0, -1.0)	# Top Right Of The Texture and Quad
        ##	glTexCoord2f(1.0, 1.0); glVertex3f( 1.0,  1.0, -1.0)	# Top Left Of The Texture and Quad
        ##	glTexCoord2f(1.0, 0.0); glVertex3f( 1.0, -1.0, -1.0)	# Bottom Left Of The Texture and Quad
        ##	glEnd();						# Done Drawing The face
        
        # Top Face
        glBindTexture(GL_TEXTURE_2D, self.textures[3])
        glBegin(GL_QUADS)				# Start Drawing The Cube
        glTexCoord2f(1.0, 1.0); glVertex3f(-hw,  hh, -hz)	# Top Left Of The Texture and Quad
        glTexCoord2f(0.0, 1.0); glVertex3f(-hw,  hh,  hz)	# Bottom Left Of The Texture and Quad
        glTexCoord2f(0.0, 0.0); glVertex3f( hw,  hh,  hz)	# Bottom Right Of The Texture and Quad
        glTexCoord2f(1.0, 0.0); glVertex3f( hw,  hh, -hz)	# Top Right Of The Texture and Quad
        glEnd();						# Done Drawing The face
        
        # Bottom Face	   
        glBindTexture(GL_TEXTURE_2D, self.textures[1])
        glBegin(GL_QUADS)				# Start Drawing The Cube
        glTexCoord2f(1.0, 1.0); glVertex3f(-hw, -hh, -hz)	# Top Right Of The Texture and Quad
        glTexCoord2f(1.0, 0.0); glVertex3f( hw, -hh, -hz)	# Top Left Of The Texture and Quad
        glTexCoord2f(0.0, 0.0); glVertex3f( hw, -hh,  hz)	# Bottom Left Of The Texture and Quad
        glTexCoord2f(0.0, 1.0); glVertex3f(-hw, -hh,  hz)	# Bottom Right Of The Texture and Quad
        glEnd();						# Done Drawing The face
        
        # Right face
        glBindTexture(GL_TEXTURE_2D, self.textures[2])
        glBegin(GL_QUADS)				# Start Drawing The Cube
        glTexCoord2f(1.0, 0.0); glVertex3f( hw, -hh, -hz)	# Bottom Right Of The Texture and Quad
        glTexCoord2f(1.0, 1.0); glVertex3f( hw,  hh, -hz)	# Top Right Of The Texture and Quad
        glTexCoord2f(0.0, 1.0); glVertex3f( hw,  hh,  hz)	# Top Left Of The Texture and Quad
        glTexCoord2f(0.0, 0.0); glVertex3f( hw, -hh,  hz)	# Bottom Left Of The Texture and Quad
        glEnd();						# Done Drawing The face
        
        # Left Face
        glBindTexture(GL_TEXTURE_2D, self.textures[4])
        glBegin(GL_QUADS)				# Start Drawing The Cube
        glTexCoord2f(1.0, 0.0); glVertex3f(-hw, -hh, -hz)	# Bottom Left Of The Texture and Quad
        glTexCoord2f(0.0, 0.0); glVertex3f(-hw, -hh,  hz)	# Bottom Right Of The Texture and Quad
        glTexCoord2f(0.0, 1.0); glVertex3f(-hw,  hh,  hz)	# Top Right Of The Texture and Quad
        glTexCoord2f(1.0, 1.0); glVertex3f(-hw,  hh, -hz)	# Top Left Of The Texture and Quad
        glEnd();						# Done Drawing The face

    def wxSize(self, event = None):
        """Called when the window is resized"""
        self.w,self.h = self.GetClientSizeTuple()
        glViewport(0, 0, self.w, self.h)		# Reset The Current Viewport And Perspective Transformation
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(self.w)/float(self.h), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def wxEraseBackground(self, event):
        """Routine does nothing, but prevents flashing"""
        pass

    def wxPaint(self, event=None):
        """Called on a paint event.

        This sets the painting drawing context, then calls the base routine wxRedrawGL()"""
        dc = wxPaintDC(self)
        self.wxRedrawGL(event)

    def wxRedraw(self, event=None):
        """Called on a redraw request

        This sets the drawing context, then calls the base routine wxRedrawGL(). It can be called by the user when a refresh is needed"""
        dc = wxClientDC(self)
        self.wxRedrawGL(event)

    def wxRedrawGL(self, event=None):
        """This is the routine called when drawing actually takes place.

        It needs to be separate so that it can be called by both paint events and by other events. It should not be called directly"""

        self.SetCurrent()

        glClear(GL_COLOR_BUFFER_BIT)
        glClear(GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)

        glPushMatrix()
        self.DrawGL()               # Actually draw here
        glPopMatrix()
        glFlush()                   # Flush
        self.SwapBuffers()  # Swap buffers

        if event: event.Skip()  # Pass event up

    def OnChar(self,event):
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
        self.wxRedraw()

        if key == ord('q'):
            self.parent.Destroy()

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
        if self.kwargs.has_key('title'):
            title = self.kwargs['title']
        else:
            title = 'Hypercube'
        frame = wxFrame(NULL, -1, title, wxDefaultPosition, wxSize(400,400))
        win = WxHypercubeWindow(*([self.data] + [frame, -1, wxPoint(5,5), wxSize(190,190)]), **self.kwargs)
        return frame

#-----------------------------------------------------

def go():
    import Spectral
##    hsi = Spectral.image('torr/torr_1.img')
    hsi = Spectral.image('92AV3C')
    hypercube(hsi, bands = [10, 20, 30])
    
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
    

