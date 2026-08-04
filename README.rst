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

Spectral Python (SPy) is a pure Python module for processing hyperspectral image
data (imaging spectroscopy data). It has functions for reading, displaying,
manipulating, and classifying hyperspectral imagery. Full details about the
package are on the `web site <https://www.spectralpython.net/>`_.


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

To run the suite of unit tests, you must have `numpy` and `pytest` installed
and you must have the `sample data files <http://spectralpython.net/user_guide_intro.html>`_
downloaded to the current directory (or one specified by the `SPECTRAL_DATA`
environment variable).

The suite also covers SPy's optional graphics/GUI functionality (2D display
via `matplotlib`/`Pillow`, and 3D display via `PySide6`/`PyOpenGL`), marked
with `graphics` (and, for the 3D window tests specifically, `gui3d`). Tests
for a given package are skipped automatically if that package isn't
installed; pass `--require-optional-deps` to turn a missing optional
dependency into a hard failure instead of a skip. A few examples:

Run only the basic tests (I/O and algorithms), explicitly excluding all
graphics/GUI tests:

.. code::

    pytest -m "not graphics" spectral/tests

Run the basic tests plus any additional graphics/GUI tests supported by
whatever optional packages are currently installed, letting tests for
anything still missing skip automatically:

.. code::

    pytest spectral/tests

Run the basic tests and force the 2D rendering/GUI tests (`matplotlib`,
`Pillow`) to run, failing rather than skipping if either package is
missing:

.. code::

    pytest --require-optional-deps=matplotlib,PIL spectral/tests

Run every available test, including the 3D (`PySide6`/`PyOpenGL`) GUI
tests, failing rather than skipping if any optional dependency -- or a
working OpenGL rendering surface -- is unavailable. If no real display is
available, run under a virtual one via `xvfb-run`:

.. code::

    xvfb-run -a pytest --require-optional-deps=all spectral/tests

Dependencies
============
Using SPy interactively with its visualization capabilities requires `IPython` and
several other packages (depending on the features used). See the
`web site <http://spectralpython.net>`_ for details.

