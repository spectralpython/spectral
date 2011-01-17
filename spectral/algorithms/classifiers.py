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
'''
Base classes for classifiers and implementations of basic statistical classifiers.
'''

import numpy

class Classifier:
    '''
    Base class for Classifiers.  Child classes must implement the
    classifySpectrum method.
    '''
    def __init__(self):
        pass
    def classifySpectrum(self, *args, **kwargs):
        raise NotImplementedError('Classifier.classifySpectrum must be overridden by a child class.')
    def classifyImage(self, image):
	'''Classifies an entire image, returning a classification map.
	
	Arguments:
	
	    `image` (ndarray or :class:`spectral.Image`)
	    
		The `MxNxB` image to classify.
	
	Returns (ndarray):
	
	    An `MxN` ndarray of integers specifying the class for each pixel.
	'''
        from spectral import status
        from algorithms import ImageIterator
        from numpy import zeros
        status.displayPercentage('Classifying image...')
        it = ImageIterator(image)
        classMap = zeros(image.shape[:2])
        N = it.getNumElements()
        i, inc = (0, N / 100)
        for spectrum in it:
            classMap[it.row, it.col] = self.classifySpectrum(spectrum)
            i += 1
            if not i % inc:
                status.updatePercentage(float(i) / N * 100.)
        status.endPercentage()
        return classMap

class SupervisedClassifier(Classifier):
    def __init__(self):
        pass
    def train(self):
        pass

class GaussianClassifier(SupervisedClassifier):
    '''A Gaussian Maximum Likelihood Classifier'''
    def __init__(self, trainingData = None, minSamples = None):
	'''Creates the classifier and optionally trains it with training data.
	
	Arguments:
	
	    `trainingData` (:class:`~spectral.algorithms.TrainingClassSet`) [default None]:
	    
		 The training classes on which to train the classifier.
	    
	    `minSamples` (int) [default None]:
	    
		Minimum number of samples required from a training class to
		include it in the classifier.
	
	'''
        if minSamples:
            self.minSamples = minSamples
        else:
            self.minSamples = None
        if trainingData:
            self.train(trainingData)

    def train(self, trainingData):
	'''Trains the classifier on the given training data.
	
	Arguments:
	
	    `trainingData` (:class:`~spectral.algorithms.TrainingClassSet`):
	    
		Data for the training classes.
	'''
        from algorithms import logDeterminant
        if not self.minSamples:
            # Set minimum number of samples to the number of bands in the image
            self.minSamples = trainingData.numBands
        self.classes = []
        for cl in trainingData:
            if cl.size() >= self.minSamples:
                self.classes.append(cl)
            else:
                print '  Omitting class %3d : only %d samples present' % (cl.index, cl.size())
        for cl in self.classes:
            if not hasattr(cl, 'stats'):
                cl.calcStatistics()
            if not hasattr(cl.stats, 'invCov'):
                cl.stats.invCov = numpy.linalg.inv(cl.stats.cov)
                cl.stats.logDetCov = logDeterminant(cl.stats.cov)

    def classifySpectrum(self, x):
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

        maxProb = -100000000000.
        maxClass = -1
        first = True

        for cl in self.classes:
            delta = (x - cl.stats.mean)[:, NewAxis]
            prob = log(cl.classProb) - 0.5 * cl.stats.logDetCov		\
                    - 0.5 * dot(transpose(delta), dot(cl.stats.invCov, delta))
            if first or prob[0,0] > maxProb:
                first = False
                maxProb = prob[0,0]
                maxClass = cl.index
        return maxClass

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
        numSamples = 0
        for cl in self.classes:
            covariance += cl.stats.numSamples * cl.stats.cov
            numSamples += cl.stats.numSamples
        self.invCovariance = numpy.linalg.inv(covariance / numSamples)

    def classifySpectrum(self, x):
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
 
        maxClass = -1
        d2_min = -1
        first = True

        for cl in self.classes:
            delta = (x - cl.stats.mean)[:, NewAxis]
            d2 = dot(transpose(delta), dot(self.invCovariance, delta))
            if first or d2 < d2_min:
                first = False
                d2_min = d2
                maxClass = cl.index
        return maxClass

