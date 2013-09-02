#########################################################################
#
#   classifiers.py - This file is part of the Spectral Python (SPy)
#   package.
#
#   Copyright (C) 2001-2010 Thomas Boggs
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
'''Base classes for classifiers and basic classifiers.'''

import numpy

from exceptions import DeprecationWarning
from warnings import warn


class Classifier:
    '''
    Base class for Classifiers.  Child classes must implement the
    classify_spectrum method.
    '''
    def __init__(self):
        pass

    def classify_spectrum(self, *args, **kwargs):
        raise NotImplementedError('Classifier.classify_spectrum must be '
                                  'overridden by a child class.')

    def classify_image(self, image):
        '''Classifies an entire image, returning a classification map.

        Arguments:

            `image` (ndarray or :class:`spectral.Image`)

                The `MxNxB` image to classify.

        Returns (ndarray):

            An `MxN` ndarray of integers specifying the class for each pixel.
        '''
        import spectral
        from algorithms import ImageIterator
        from numpy import zeros
        status = spectral._status
        status.display_percentage('Classifying image...')
        it = ImageIterator(image)
        class_map = zeros(image.shape[:2])
        N = it.get_num_elements()
        i, inc = (0, N / 100)
        for spectrum in it:
            class_map[it.row, it.col] = self.classify_spectrum(spectrum)
            i += 1
            if not i % inc:
                status.update_percentage(float(i) / N * 100.)
        status.end_percentage()
        return class_map

    #-------------------
    # Deprecated methods
    #-------------------
    def classifySpectrum(self, *args, **kwargs):
        warn('Classifier.classifySpectrum has been deprecated. '
             + 'Use Classifier.classify_spectrum.', DeprecationWarning)
        return self.classifySpectrum(*args, **kwargs)

    def classifyImage(self, image):
        warn('Classifier.addClass has been deprecated. '
             + 'Use Classifier.classify_image.', DeprecationWarning)
        return self.classify_image(image)


class SupervisedClassifier(Classifier):
    def __init__(self):
        pass

    def train(self):
        pass


class GaussianClassifier(SupervisedClassifier):
    '''A Gaussian Maximum Likelihood Classifier'''
    def __init__(self, training_data=None, min_samples=None):
        '''Creates the classifier and optionally trains it with training data.

        Arguments:

            `training_data` (:class:`~spectral.algorithms.TrainingClassSet`):

                 The training classes on which to train the classifier.

            `min_samples` (int) [default None]:

                Minimum number of samples required from a training class to
                include it in the classifier.

        '''
        if min_samples:
            self.min_samples = min_samples
        else:
            self.min_samples = None
        if training_data:
            self.train(training_data)

    def train(self, training_data):
        '''Trains the classifier on the given training data.

        Arguments:

            `training_data` (:class:`~spectral.algorithms.TrainingClassSet`):

                Data for the training classes.
        '''
        from algorithms import log_det
        if not self.min_samples:
            # Set minimum number of samples to the number of bands in the image
            self.min_samples = training_data.nbands
        self.classes = []
        for cl in training_data:
            if cl.size() >= self.min_samples:
                self.classes.append(cl)
            else:
                print '  Omitting class %3d : only %d samples present' % (
                    cl.index, cl.size())
        for cl in self.classes:
            if not hasattr(cl, 'stats'):
                cl.calc_stats()
            if not hasattr(cl.stats, 'inv_cov'):
                cl.stats.inv_cov = numpy.linalg.inv(cl.stats.cov)
                cl.stats.log_det_cov = log_det(cl.stats.cov)

    def classify_spectrum(self, x):
        '''
        Classifies a pixel into one of the trained classes.

        Arguments:

            `x` (list or rank-1 ndarray):

                The unclassified spectrum.

        Returns:

            `classIndex` (int):

                The index for the :class:`~spectral.algorithms.TrainingClass`
                to which `x` is classified.
        '''
        from numpy import dot, transpose
        from numpy.oldnumeric import NewAxis
        from math import log

        max_prob = -100000000000.
        max_class = -1
        first = True

        for cl in self.classes:
            delta = (x - cl.stats.mean)[:, NewAxis]
            prob = log(cl.class_prob) - 0.5 * cl.stats.log_det_cov \
                - 0.5 * dot(transpose(delta), dot(cl.stats.inv_cov, delta))
            if first or prob[0, 0] > max_prob:
                first = False
                max_prob = prob[0, 0]
                max_class = cl.index
        return max_class

    def classifySpectrum(self, *args, **kwargs):
        warn('GaussianClassifier.classifySpectrum has been deprecated. '
             + 'Use GaussianClassifier.classify_spectrum.', DeprecationWarning)
        return self.classifySpectrum(*args, **kwargs)


class MahalanobisDistanceClassifier(GaussianClassifier):
    '''A Classifier using Mahalanobis distance for class discrimination'''
    def train(self, trainingData):
        '''Trains the classifier on the given training data.

        Arguments:

            `trainingData` (:class:`~spectral.algorithms.TrainingClassSet`):

                Data for the training classes.
        '''
        GaussianClassifier.train(self, trainingData)

        covariance = numpy.zeros(self.classes[0].stats.cov.shape, numpy.float)
        nsamples = 0
        for cl in self.classes:
            covariance += cl.stats.nsamples * cl.stats.cov
            nsamples += cl.stats.nsamples
        self.inv_cov = numpy.linalg.inv(covariance / nsamples)

    def classify_spectrum(self, x):
        '''
        Classifies a pixel into one of the trained classes.

        Arguments:

            `x` (list or rank-1 ndarray):

                The unclassified spectrum.

        Returns:

            `classIndex` (int):

                The index for the :class:`~spectral.algorithms.TrainingClass`
                to which `x` is classified.
        '''
        from numpy import dot, transpose
        from numpy.oldnumeric import NewAxis

        max_class = -1
        d2_min = -1
        first = True

        for cl in self.classes:
            delta = (x - cl.stats.mean)[:, NewAxis]
            d2 = dot(transpose(delta), dot(self.inv_cov, delta))
            if first or d2 < d2_min:
                first = False
                d2_min = d2
                max_class = cl.index
        return max_class

    def classifySpectrum(self, *args, **kwargs):
        warn('MahalanobisDistanceClassifier.classifySpectrum has been '
             + 'deprecated. Use '
             + 'MahalanobisDistanceClassifier.classify_spectrum.',
             DeprecationWarning)
        return self.classify_pectrum(*args, **kwargs)
