#########################################################################
#
#   SpyGnuplot.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2001 Thomas Boggs
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
A module to use Gnuplot for creating x-y plots of pixel spectra.
'''

import Gnuplot
from Numeric import *

def plot(data):
    '''
    Creates an x-y plot.

    USAGE: plot(data)

    If data is a vector, all the values in data will be drawn in a
    single series. If data is a 2D array, each column of data will
    be drawn as a separate series.
    '''

    g = Gnuplot.Gnuplot()
    g('set data style lines')
    g('set grid')
    s = shape(data)

    if len(s) == 1:
        # plot a vector
        g.plot(Gnuplot.Data(range(s[0]), data, with='lines'))
    elif len(s) == 2:
        xvals = range(s[1])
        g.plot(Gnuplot.Data(xvals, data[0,:], with='lines'))
        for i in range(1, s[0]):
            g.replot(Gnuplot.Data(xvals, data[i,:], with='lines'))
    wait()

qp = plot
        

if __name__ == '__main__':
    '''This was just for debugging and should not be here now.'''
    g = Gnuplot.Gnuplot(debug=1)
    g('set data style lines')
    g.title('A simple example') # (optional)
    g('set data style linespoints') # give gnuplot an arbitrary command
    # Plot a list of (x, y) pairs (tuples or a Numeric array would
    # also be OK):
    #g.plot([[0,1.1], [1,5.8], [2,3.3], [3,4.2]])
    x = array(range(100)) * 0.1
    y = sin(x)
    z = cos(x)
    ds = Gnuplot.Data(x, y, title = 'sin(x)', with = 'lines')
    dc = Gnuplot.Data(x, z, title = 'cos(x)')
    g.xlabel('Channel')
    g.ylabel('Intensity')
    g.plot(ds)
    g.replot(dc)
