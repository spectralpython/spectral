'''
Top-level functions & classes.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numbers
import numpy as np
import pickle
import os
from warnings import warn

#from .algorithms.algorithms import TrainingClassSet
#from . import io
#from .io import aviris, envi, erdas, spyfile
#from .io.spyfile import find_file_path, SpyFile

from . import settings

def _init():
    '''Basic configuration of the spectral package.'''
    _setup_logger()
    try:
        global settings
        from .graphics import graphics as spygraphics
        from .graphics import spypylab
        settings.plotter = spypylab
        settings.viewer = spygraphics
    except:
        raise
        warn('Unable to import or configure pylab plotter.  Spectrum plots '
             'will be unavailable.', UserWarning)

    from .utilities import status
    spectral = __import__(__name__.split('.')[0])
    spectral._status = status.StatusDisplay()

def _setup_logger():
    logger = logging.getLogger('spectral')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s:%(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class BandInfo:
    '''A BandInfo object characterizes the spectral bands associated with an
    image. All BandInfo member variables are optional.  For *N* bands, all
    members of type <list> will have length *N* and contain float values.

    =================   =====================================   =======
        Member                  Description                     Default
    =================   =====================================   =======
    centers             List of band centers                    None
    bandwidths          List of band FWHM values                None
    centers_stdevs      List of std devs of band centers        None
    bandwidth_stdevs    List of std devs of bands FWHMs         None
    band_quantity       Image data type (e.g., "reflectance")   ""
    band_unit           Band unit (e.g., "nanometer")           ""
    =================   =====================================   =======
    '''
    def __init__(self):
        self.centers = None
        self.bandwidths = None
        self.centers_stdevs = None
        self.bandwidth_stdevs = None
        self.band_quantity = None
        self.band_unit = None


def open_image(file):
    '''
    Locates & opens the specified hyperspectral image.

    Arguments:

        file (str):
            Name of the file to open.

    Returns:

        SpyFile object to access the file.

    Raises:

        IOError.

    This function attempts to determine the associated file type and open the
    file. If the specified file is not found in the current directory, all
    directories listed in the :const:`SPECTRAL_DATA` environment variable will
    be searched until the file is found.  If the file being opened is an ENVI
    file, the `file` argument should be the name of the header file.
    '''
    from . import io
    pathname = io.spyfile.find_file_path(file)

    # Try to open it as an ENVI header file.
    try:
        return io.envi.open(pathname)
    except io.envi.FileNotAnEnviHeader:
        # It isn't an ENVI file so try another file type
        pass
    except:
        raise

    # Maybe it's an Erdas Lan file
    try:
        return io.erdas.open(pathname)
    except:
        pass

    # See if the size is consistent with an Aviris file
    try:
        return io.aviris.open(pathname)
    except:
        pass

    raise IOError('Unable to determine file type or type not supported.')


def load_training_sets(file, image=None):
    '''
    Loads a list of TrainingSet objects from a file.  This function assumes
    that all the sets in the list refer to the same image and mask array.
    If that is not the case, this function should not be used.
    '''
    from .algorithms.algorithms import TrainingClassSet
    ts = TrainingClassSet()
    ts.load(file, image)
    return ts

