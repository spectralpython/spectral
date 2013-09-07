#!/usr/bin/env python

try:
    from setuptools import setup
except:
    from distutils.core import setup

long_description = '''
Spectral Python (SPy) is a pure Python module for processing hyperspectral
image data. SPy has functions for reading, displaying, manipulating, and
classifying hyperspectral imagery. SPy can be used interactively from the
Python command prompt or via Python scripts. SPy is free, open source software
distributed under the GNU General Public License.'''

setup(name='spectral',
      version='0.12',
      description='Spectral Python (SPy) is a Python module for hyperspectral image processing.',
      long_description=long_description,
      author='Thomas Boggs',
      author_email='tboggs@users.sourceforge.net',
      license='GPL',
      url='http://spectralpython.sourceforge.net',
      download_url='https://sourceforge.net/projects/spectralpython/files/',
      packages=['spectral', 'spectral.algorithms', 'spectral.database',
                'spectral.graphics', 'spectral.io', 'spectral.tests',
                'spectral.utilities'],
      platforms=['Platform-Independent'],
      classifiers=[
	'Development Status :: 4 - Beta',
	'Intended Audience :: Science/Research',
	'License :: OSI Approved :: GNU General Public License (GPL)',
	'Operating System :: OS Independent',
	'Programming Language :: Python :: 2.6',
	'Programming Language :: Python :: 2.7'
      ]
     )
