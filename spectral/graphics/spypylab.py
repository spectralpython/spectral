#########################################################################
#
#   spypylab.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2001-2010 Thomas Boggs
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
