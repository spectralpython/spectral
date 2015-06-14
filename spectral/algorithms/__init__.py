#########################################################################
#
#   algorithms/__init__.py - This file is part of the Spectral Python
#  (SPy) package.
#
#   Copyright (C) 2001-2006 Thomas Boggs
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

from __future__ import division, print_function, unicode_literals

from .algorithms import (mean_cov, covariance, principal_components, bdist,
                        linear_discriminant, create_training_classes, ndvi,
                        orthogonalize, transform_image, unmix, spectral_angles,
                        calc_stats, cov_avg, msam, noise_from_diffs, mnf,
                        GaussianStats, ppi)
from .classifiers import *
from .clustering import L1, L2, kmeans
from .resampling import BandResampler
from .transforms import LinearTransform
from .detectors import *
from .spatial import *
