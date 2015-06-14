#!/usr/bin/env python

import ast
import re
try:
    from setuptools import setup
except:
    from distutils.core import setup


# taken from Flask
_version_re = re.compile(r'__version__\s+=\s+(.*)')
with open('spectral/__init__.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))

long_description = '''Spectral Python (SPy) is a pure Python module for
processing hyperspectral image data (imaging spectroscopy data). It has
functions for reading, displaying, manipulating, and classifying hyperspectral
imagery. SPy is Free, Open Source Software (FOSS) distributed under the GNU
General Public License.'''

setup(name='spectral',
      version=version,
      description='Spectral Python (SPy) is a Python module for hyperspectral image processing.',
      long_description=long_description,
      author='Thomas Boggs',
      author_email='thomas.boggs@gmail.com',
      license='GPL',
      url='http://spectralpython.net',
      download_url='https://github.com/spectralpython/spectral/releases/latest',
      packages=['spectral', 'spectral.algorithms', 'spectral.database',
                'spectral.graphics', 'spectral.io', 'spectral.tests',
                'spectral.utilities'],
      platforms=['Platform-Independent'],
      install_requires=['numpy'],
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
