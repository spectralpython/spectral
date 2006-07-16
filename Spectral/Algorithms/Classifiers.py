#########################################################################
#
#   Classifiers.py - This file is part of the Spectral Python (SPy)
#   package.
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
'''
Base classes for classifiers and implementations of basic statistical classifiers.
'''

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
        from Spectral import status, Int0
        from Algorithms import ImageIterator
        from Numeric import zeros
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
        if minSamples:
            self.minSamples = minSamples
        else:
            self.minSamples = None
        if trainingData:
            self.train(trainingData)
    def train(self, trainingData):
        from LinearAlgebra import inverse
        from Algorithms import logDeterminant
        if not self.minSamples:
            # Set minimum number of samples to the number of bands in the image
            self.minSamples = trainingData[0].image.shape[2]
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
                cl.stats.invCov = inverse(cl.stats.cov)
                cl.stats.logDetCov = logDeterminant(cl.stats.cov)

    def classifySpectrum(self, x):
        '''
        Classify pixel into one of classes.

        USAGE: classIndex = classifier.classifySpectrum(x)

        ARGUMENTS:
            x           The spectrum to classify
        RETURN VALUE
            classIndex  The 'index' property of the most likely class
                        in classes.
        '''
        from Numeric import NewAxis, matrixmultiply, transpose
        from math import log

        maxProb = -100000000000.
        maxClass = -1
        first = True

        for cl in self.classes:
            delta = (x - cl.stats.mean)[:, NewAxis]
            prob = log(cl.classProb) - 0.5 * cl.stats.logDetCov		\
                    - 0.5 * matrixmultiply(transpose(delta),	\
                    matrixmultiply(cl.stats.invCov, delta))
            if first or prob[0,0] > maxProb:
                first = False
                maxProb = prob[0,0]
                maxClass = cl.index
        return maxClass

class MahalanobisDistanceClassifier(GaussianClassifier):
    '''A Classifier using Mahalanobis distance for class discrimination'''
    def train(self, trainingData):
        '''
        Calculate a single coveriance as a weighted average of the
        individual training class covariances.
        '''
        import Numeric
        from LinearAlgebra import inverse
        GaussianClassifier.train(self, trainingData)

        covariance = Numeric.zeros(self.classes[0].stats.cov.shape, Numeric.Float)
        numSamples = 0
        for cl in self.classes:
            covariance += cl.stats.numSamples * cl.stats.cov
            numSamples += cl.stats.numSamples
        self.invCovariance = inverse(covariance / numSamples)

    def classifySpectrum(self, x):
        '''
        Classify pixel based on minimum Mahalanobis distance.

        USAGE: classIndex = classifier.classifySpectrum(x)

        ARGUMENTS:
            x           The spectrum to classify
        RETURN VALUE
            classIndex  The 'index' property of the most likely class
                        in classes.
        '''
        from Numeric import NewAxis, matrixmultiply, transpose
 
        maxClass = -1
        d2_min = -1
        first = True

        for cl in self.classes:
            delta = (x - cl.stats.mean)[:, NewAxis]
            d2 = matrixmultiply(transpose(delta), matrixmultiply(self.invCovariance, delta))
            if first or d2 < d2_min:
                first = False
                d2_min = d2
                maxClass = cl.index
        return maxClass

