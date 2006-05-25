
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
        classMap = zeros(image.shape[:2], Int0)
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
            classes     A list of TrainingSet objects with stats
        RETURN VALUE
            classIndex  The 'index' property of the most likely class
                        in classes.
        '''
        from Numeric import NewAxis, matrixmultiply, transpose
        from math import log

        maxProb = -100000000000.
        maxClass = -1

        for i in range(len(self.classes)):
            cl = self.classes[i]
            delta = (x - cl.stats.mean)[:, NewAxis]
            prob = log(cl.classProb) - 0.5 * cl.stats.logDetCov		\
                    - 0.5 * matrixmultiply(transpose(delta),	\
                    matrixmultiply(cl.stats.invCov, delta))
            if i == 0:
                maxProb = prob[0,0]
                maxClass = self.classes[0].index
            elif (prob[0,0] > maxProb):
                maxProb = prob[0,0]
                maxClass = self.classes[i].index
        return maxClass
