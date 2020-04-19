'''
Functions for resampling a spectrum from one band discretization to another.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import numpy as np

from ..spectral import BandInfo

def erf_local(x):
    # save the sign of x
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)
    return sign*y # erf(-x) = -erf(x)

try:
    from math import erf
except:
    try:
        from scipy.special import erf
    except:
        erf = erf_local

def erfc(z):
    '''Complement of the error function.'''
    return 1.0 - erf(z)

def normal_cdf(x):
    '''CDF of the normal distribution.'''
    sqrt2 = 1.4142135623730951
    return 0.5 * erfc(-x / sqrt2)

def normal_integral(a, b):
    '''Integral of the normal distribution from a to b.'''
    return normal_cdf(b) - normal_cdf(a)

def ranges_overlap(R1, R2):
    '''Returns True if there is overlap between ranges of pairs R1 and R2.'''
    if (R1[0] < R2[0] and R1[1] < R2[0]) or \
       (R1[0] > R2[1] and R1[1] > R2[1]):
        return False
    return True

def overlap(R1, R2):
    '''Returns (min, max) of overlap between the ranges of pairs R1 and R2.'''
    return (max(R1[0], R2[0]), min(R1[1], R2[1]))

def normal(mean, stdev, x):
    sqrt_2pi = 2.5066282746310002
    return math.exp(-((x - mean) / stdev)**2 / 2.0) / (sqrt_2pi * stdev)

def build_fwhm(centers):
    '''Returns FWHM list, assuming FWHM is midway between adjacent bands.
    '''
    fwhm = [0] * len(centers)
    fwhm[0] = centers[1] - centers[0]
    fwhm[-1] = centers[-1] - centers[-2]
    for i in range(1, len(centers) - 1):
        fwhm[i] = (centers[i + 1] - centers[i - 1]) / 2.0
    return fwhm

def create_resampling_matrix(centers1, fwhm1, centers2, fwhm2):
    '''
    Returns a resampling matrix to convert spectra from one band discretization
    to another.  Arguments are the band centers and full-width half maximum
    spectral response for the original and new band discretizations.
    '''
    logger = logging.getLogger('spectral')

    sqrt_8log2 = 2.3548200450309493

    N1 = len(centers1)
    N2 = len(centers2)
    bounds1 = [[centers1[i] - fwhm1[i] / 2.0, centers1[i] + fwhm1[i] /
                2.0] for i in range(N1)]
    bounds2 = [[centers2[i] - fwhm2[i] / 2.0, centers2[i] + fwhm2[i] /
                2.0] for i in range(N2)]

    M = np.zeros([N2, N1])

    jStart = 0
    nan = float('nan')
    for i in range(N2):
        stdev = fwhm2[i] / sqrt_8log2
        j = jStart

        # Find the first original band that overlaps the new band
        while j < N1 and bounds1[j][1] < bounds2[i][0]:
            j += 1

        if j == N1:
            logger.info(('No overlap for target band %d (%f / %f)' % (
                i, centers2[i], fwhm2[i])))
            M[i, 0] = nan
            continue

        matches = []

        # Get indices for all original bands that overlap the new band
        while j < N1 and bounds1[j][0] < bounds2[i][1]:
            if ranges_overlap(bounds1[j], bounds2[i]):
                matches.append(j)
            j += 1

        # Put NaN in first element of any row that doesn't produce a band in
        # the new schema.
        if len(matches) == 0:
            logger.info('No overlap for target band %d (%f / %f)',
                         i, centers2[i], fwhm2[i])
            M[i, 0] = nan
            continue

        # Determine the weights for the original bands that overlap the new
        # band. There may be multiple bands that overlap or even just a single
        # band that only partially overlaps.  Weights are normoalized so either
        # case can be handled.

        overlaps = [overlap(bounds1[k], bounds2[i]) for k in matches]
        contribs = np.zeros(len(matches))
        A = 0.
        for k in range(len(matches)):
            #endNorms = [normal(centers2[i], stdev, x) for x in overlaps[k]]
            #dA = (overlaps[k][1] - overlaps[k][0]) * sum(endNorms) / 2.0
            (a, b) = [(x - centers2[i]) / stdev for x in overlaps[k]]
            dA = normal_integral(a, b)
            contribs[k] = dA
            A += dA
        contribs = contribs / A
        for k in range(len(matches)):
            M[i, matches[k]] = contribs[k]
    return M

class BandResampler:
    '''A callable object for resampling spectra between band discretizations.

    A source band will contribute to any destination band where there is
    overlap between the FWHM of the two bands.  If there is an overlap, an
    integral is performed over the region of overlap assuming the source band
    data value is constant over its FWHM (since we do not know the true
    spectral load over the source band) and the destination band has a Gaussian
    response function. Any target bands that do not have any overlapping source
    bands will contain NaN as the resampled band value.

    If bandwidths are not specified for source or destination bands, the bands
    are assumed to have FWHM values that span half the distance to the adjacent
    bands.
    '''
    def __init__(self, centers1, centers2, fwhm1=None, fwhm2=None):
        '''BandResampler constructor.

        Usage:

            resampler = BandResampler(bandInfo1, bandInfo2)

            resampler = BandResampler(centers1, centers2, [fwhm1 = None [, fwhm2 = None]])

        Arguments:

            `bandInfo1` (:class:`~spectral.BandInfo`):

                Discretization of the source bands.

            `bandInfo2` (:class:`~spectral.BandInfo`):

                Discretization of the destination bands.

            `centers1` (list):

                floats defining center values of source bands.

            `centers2` (list):

                floats defining center values of destination bands.

            `fwhm1` (list):

                Optional list defining FWHM values of source bands.

            `fwhm2` (list):

                Optional list defining FWHM values of destination bands.

        Returns:

            A callable BandResampler object that takes a spectrum corresponding
            to the source bands and returns the spectrum resampled to the
            destination bands.

        If bandwidths are not specified, the associated bands are assumed to
        have FWHM values that span half the distance to the adjacent bands.
        '''
        if isinstance(centers1, BandInfo):
            fwhm1 = centers1.bandwidths
            centers1 = centers1.centers
        if isinstance(centers2, BandInfo):
            fwhm2 = centers2.bandwidths
            centers2 = centers2.centers
        if fwhm1 is None:
            fwhm1 = build_fwhm(centers1)
        if fwhm2 is None:
            fwhm2 = build_fwhm(centers2)
        self.matrix = create_resampling_matrix(
            centers1, fwhm1, centers2, fwhm2)

    def __call__(self, spectrum):
        '''Takes a source spectrum as input and returns a resampled spectrum.

        Arguments:

            `spectrum` (list or :class:`numpy.ndarray`):

                list or vector of values to be resampled.  Must have same
                length as the source band discretiation used to created the
                resampler.

        Returns:

            A resampled rank-1 :class:`numpy.ndarray` with length corresponding
            to the destination band discretization used to create the resampler.

        Any target bands that do not have at lease one overlapping source band
        will contain `float('nan')` as the resampled band value.'''
        return np.dot(self.matrix, spectrum)
