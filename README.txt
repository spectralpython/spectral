================================================================================
Installation Instructions
================================================================================

To install SPy, unpack the source distribution archive, `cd` into the
directory created when the archive is unpacked (e.g., "spectral.x.y"), and
type the following::

    python setup.py install

================================================================================
SPy 0.12
================================================================================
Release date: 2013.09.06

New Featues
-----------

* Added a wrapper around matplotlib's `imshow` to easily display HSI data.

* A new memmap interface is provided for SpyFile objects
  (via the `open_memmap` method), including the ability to open writable
  memmaps.

* Algorithm progress display can be enabled/disabled via the settings object.

* RX anomaly detection can be performed using local statistics by specifying
  an inner/outer window.

* A custom background color can be specified when calling `view_cube`.

* Summary statistics are printed for unit test execution.

Changes
-------

* `get_image_display_data` has been renamed `get_rgb`.

* `view_cube` will also accept an ndaray as the "top" keyword.

* If present, image band info is saved when `envi.save_image` is called.

* Allow calling :func:`~spectral.oi.envi.create_image` using keyword args
  instead of ENVI-specific header paramter names.

* `save_rgb` automatically determines the output file type, based on the
  filename extension.

* Results of several ImageArray methods will be cast to an ndarray.

* The Image base class is now a new-style class.


Bug Fixes
---------

* Eliminated texture-wrapping display artifact near edges of displayed image
  cubes (called via `view_cube`).

* RX.__call__ was failing when image statistics were not provided to class
  constructor.

* Applied Ferdinand Deger's bugfix for `envi.create_image`.



================================================================================
SPy 0.11
================================================================================
Release date: 2013.04.03

New Featues
-----------

* RX anomaly detector.

* Ability to save and create images in ENVI format.

* Added `GaussianStats` class (returned by `calc_stats`). This class can be
  transformed by a `LinearTransform`.  It has a `get_whitening_transform`
  method that returns a callable transform to whiten image data.

* Added a unit-testing sub-package (`spectral.tests`)

Changes
-------

* Changed severals function to accept GaussianStats objects instead of
  sepaarate mean & covariance.

* Changed names of several functions for consistency:

  - `open_image` replaces `image`.

  - `save_rgb` replaces `save_image`

* Improved support for additional data types by reading byte strings into
  numpy arrays with dtypes instead of using builtin array module.

Bug Fixes
---------

* 32-bit integer image data  was not being read properly.

================================================================================
SPy 0.10.1
================================================================================
Release date: 2013.02.23

This is a bug-fix release that corrects the spectrum displayed when double-
clicking on a raster display.  Version 0.10 introduced a bug that had the
row/column swapped, resulting in either the wrong pixel being plotted or an
exception raised.

================================================================================
SPy 0.10
================================================================================
Release date: 2013.02.17

As of this release, SPy now uses IPython for non-blocking GUI windows. IPython
should be started in "--pylab" mode with the appropriate backend set (see
:ref:`starting_ipython`). The standard python interpreter can still be used if
GUI functions are not being called.

New Features
------------

* `LinearTransform` and `transform_image` now handle scalar transforms.

* All functions opening a GUI window will now return a proxy object to enable
  access to any associated data (e.g., accessing changed class values in an
  N-D data display).

* GUI functions are now aware of differences in wxWidgets versions
  (2.8.x vs. 2.9.x).

Changes
-------

* SPy no longer requires explicit creation of a new wx thread.  Instead,
  running SPy interactively with GUI functions now requires using IPython
  in "pylab" mode.

* A few functions have been renamed for consistency:

  * `hypercube` is now `view_cube`.
  
  * `ndwindow is now `view_nd`.

* numpy is used for more covariance calculations (e.g., class-specific
  covariance) to improve performance on multi-core systems.

* Two new parameters have been added to the `spectral.settings` object:

  1. `START_WX_APP` : If this parameter is True and no wx App exists when a
     GUI function is called, then an App will be started to prevent an error.
     
  2. `WX_GL_DEPTH_SIZE` : If the default GL_DEPTH_SIZE is not supported by the
     host system (resulting in a blank GLCanvas in `view_cube` or `view_nd`),
     this parameter can be reduced (e.g., to 24 or 16) to enable OpenGL
     rendering.
  
Bug Fixes
---------

* Spectral plotting failed when double-clicking a transformed image due to
  band info being unavailable.  A check is now performed to prevent this.

* OpenGL-related calls will no longer fail if GLUT or an associated font is
  not available.

================================================================================
SPy 0.9
================================================================================
Release date: 2013.01.23

- New Features

  - Added a linear target detector (MatchedFilter).

  - Added a class for linear transformations (LinearTransform).

- Changes

  - `principal_components` function now returns a object, which contains return
     values previously in a tuple, , as well as the associated linear transform,
     and a `reduce` method.
  
  - `linear_discriminant` function now returns an object, which contains return
    values previously in a tuple, as well as the associated linear transform.
  
  - Covariance calculation is now performed using 64-bit floats.

- Bug Fixes

  - Fixed a bug causing `ndwindow` to fail when no class mask is passed as a
    keyword arg.

================================================================================
SPy 0.8 Release Notes
================================================================================
Release date: 2012.07.15

- New Features

  - The :func:`~spectral.graphics.ndwindow.ndwindow` function enables viewing of
    high-dimensional images in a 3D display. See :ref:`nd_displays` for details.

- Changes

  - Hypercube display now uses mouse control for pan/zoom/rotate.

- Bug Fixes

  - Fixed a bug in several deprecation warnings that caused infinte recursion.

  - Fixed mismatch in parameter names in kmeans_ndarray.

================================================================================
SPy 0.7 Release Notes
================================================================================
Release date: 2012.02.19

- Changes

  - Changed many function/method names to be more consistent with external
    packages.  Use of most old names will generate a deprecation warning but
    some will require immediate changes to client code.

  - :func:`spectral.kmeans` runs about 10 times faster now for loaded images.


- Bug Fixes

  - The Erdas LAN file interface was modified because the previous reference
    file had mixed endian between header and data sections.  If you are using
    the old sample file "92AV3C", then start using the "92AV3C.lan" file
    available on the web site (see Intro section of the user's guide). This file
    has consistent endian-ness between header and image data sections

  - Fixed a few bugs that potentially caused problems with the BIP and BSQ file
    interfaces.  The specific methods fixed are:
    
    * BipFile.read_bands
    * BsqFile.read_subregion
    * BsqFile.read_subimage
 

================================================================================
SPy 0.6 Release Notes
================================================================================
Release date: 2011.01.17

- New Features:

  - Support for parsing & loading spectra from the ASTER Spectral Library.
  
  - Ability to save ENVI spectral library files.

  - :meth:`spectral.kmeans` will accept a :exc:`KeyboardInterrupt` exception
    (i.e., CTRL-C) and return the results as of the previous iteration.

  - Documention is now online via Sphinx.

- Changes

  - Major changes to module/sub-module layout. Biggest change is that the top-
    level module is now "spectral" instead of "Spectra" (capitalization). Many
    functions/classes have moved between files and sub-modules but that should
    be transparent to most users since the most obvious names are automatically
    imported into the top-level module namespace.

  - Additional ENVI data file extensions are now searched (.bil, .bsq, .bip,)

  - Changed default colors in :obj:`spectral.spyColors`, which is the default
    color palette used by :meth:`spectral.viewIndexed`.

  - :meth:`spectral.transformImage` is now the means to apply a linear transform
    to an image (rather than creating a :class:`spectral.TransformedImage`
    directly) because it handles both :class:`spectral.SpyFile` and
    :class:`numpy.ndarray` objects.

  - 64-bit floats are now used for covariance matrices.

  - Changed SPECTRAL_DATA path delimiter from semi-colon to colon.
    
- Bug fixes

  - Fixed a bug preventing successful reading of ENVI header files where an
    equal ("=") symbol appears in a parameter value.

  - Fixed bug where a ColorScale might return an incorrect color if the scale
    contained negative values.

  - :meth:`cluster` would fail if only 2 clusters were requested.
  
  - :meth:`kmeans` was reporting an incorrect number of pixels reassigned
    between iterations (did not affect final convergence).

  - :meth:`logDeterminant` was crashing when receiving negative values.
  
  - Missing a potential byte swap in :meth:`spectral.io.bilfileBilFile.readDatum`.


