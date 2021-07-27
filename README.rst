Spectral Python (SPy)
---------------------

.. image:: https://github.com/spectralpython/spectral/actions/workflows/python-package.yml/badge.svg?branch=master
   :target: https://github.com/spectralpython/spectral/actions/workflows/python-package.yml

.. image:: https://badges.gitter.im/spectralpython/spectral.svg
   :alt: Join the chat at https://gitter.im/spectralpython/spectral
   :target: https://gitter.im/spectralpython/spectral?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. image:: https://anaconda.org/conda-forge/spectral/badges/version.svg
   :target: https://anaconda.org/conda-forge/spectral

.. image:: https://anaconda.org/conda-forge/spectral/badges/platforms.svg
   :target: https://anaconda.org/conda-forge/spectral

.. image:: https://anaconda.org/conda-forge/spectral/badges/license.svg
   :target: https://anaconda.org/conda-forge/spectral

.. image:: https://anaconda.org/conda-forge/spectral/badges/downloads.svg
   :target: https://anaconda.org/conda-forge/spectral

.. image:: https://anaconda.org/conda-forge/spectral/badges/installer/conda.svg
   :target: https://conda.anaconda.org/conda-forge

Spectral Python (SPy) is a pure Python module for processing hyperspectral image
data (imaging spectroscopy data). It has functions for reading, displaying,
manipulating, and classifying hyperspectral imagery. Full details about the
package are on the `web site <http://spectralpython.net>`_.


Installation Instructions
=========================

The latest release is always hosted on `PyPI <https://pypi.python.org/pypi/spectral>`_,
so if you have `pip` installed, you can install SPy from the command line with

.. code::

    pip install spectral

Packaged distributions are also hosted at `PyPI <https://pypi.python.org/pypi/spectral>`_
and `GitHub <https://github.com/spectralpython/spectral/releases/latest>`_
so you can download and unpack the latest zip/tarball, then type

.. code::

    python setup.py install

To install the latest development version, download or clone the git repository
and install as above. No explicit installation is required so you can simply
access (or symlink) the `spectral` module within the source tree.

**Finally**, up-to-date guidance on how to install via the popular conda package 
and environment management system can be found at official `conda-forge documentation <https://anaconda.org/conda-forge/spectral>`_.

Unit Tests
==========

To run the suite of unit tests, you must have `numpy` installed and you must
have the `sample data files <http://spectralpython.net/user_guide_intro.html>`_
downloaded to the current directory (or one specified by the `SPECTRAL_DATA`
environment variable). To run the unit tests, type

.. code::

    python -m spectral.tests.run

Dependencies
============
Using SPy interactively with its visualization capabilities requires `IPython` and
several other packages (depending on the features used). See the
`web site <http://spectralpython.net>`_ for details.

