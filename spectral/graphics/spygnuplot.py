'''
A module to use Gnuplot for creating x-y plots of pixel spectra.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import Gnuplot

xyplot = Gnuplot.Gnuplot()


def plot(data, source=None):
    '''
    Creates an x-y plot.

    USAGE: plot(data)

    If data is a vector, all the values in data will be drawn in a
    single series. If data is a 2D array, each column of data will
    be drawn as a separate series.
    '''

    from numpy import shape
    global xyplot

    g = Gnuplot.Gnuplot()
    g('set style data lines')
    g('set grid')
    s = shape(data)

    if len(s) == 1:
        # plot a vector
        g('set xrange [0: %d]' % s[0])
        g.plot(Gnuplot.Data(list(range(s[0])), data))
    elif len(s) == 2:
        xvals = list(range(s[1]))
        g('set xrange [0: %d]' % s[1])
        g.plot(Gnuplot.Data(xvals, data[0, :]))
        for i in range(1, s[0]):
            g.replot(Gnuplot.Data(xvals, data[i, :]))
    xyplot = g
    return g
