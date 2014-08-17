#!/usr/bin/env python

try:
    from setuptools import setup
except:
    from distutils.core import setup

import spectral

long_description = '''Spectral Python (SPy) is a pure Python module for
processing hyperspectral image data (imaging spectroscopy data). It has
functions for reading, displaying, manipulating, and classifying hyperspectral
imagery. SPy is Free, Open Source Software (FOSS) distributed under the GNU
General Public License.'''

setup(name='spectral',
      version=spectral.__version__,
      description='Spectral Python (SPy) is a Python module for hyperspectral image processing.',
      long_description=long_description,
      author='Thomas Boggs',
      author_email='thomas.boggs@gmail.com',
      license='GPL',
      url='http://spectralpython.net',
      download_url='https://sourceforge.net/projects/spectralpython/files/',
      packages=['spectral', 'spectral.algorithms', 'spectral.database',
                'spectral.graphics', 'spectral.io', 'spectral.tests',
                'spectral.utilities'],
      platforms=['Platform-Independent'],
      classifiers=[	'Development Status :: 4 - Beta',
                    'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',
                    'Operating System :: OS Independent',
                    'Programming Language :: Python :: 2.6',
                    'Programming Language :: Python :: 2.7',
                    'Environment :: Console',
                    'Natural Language :: English',
                    'Intended Audience :: Science/Research',
                    'Topic :: Scientific/Engineering :: Image Recognition',
                    'Topic :: Scientific/Engineering :: GIS',
                    'Topic :: Scientific/Engineering :: Information Analysis',
                    'Topic :: Scientific/Engineering :: Visualization'
      ]
)
