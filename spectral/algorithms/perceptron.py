#########################################################################
#
#   perceptron.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2001-2008 Thomas Boggs
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
Classes and functions for classification with neural networks.
'''

import numpy

class Neuron:
    '''A neuron with a logistic sigmoid activation function.'''
    def __init__(self, k = 1.0):
        '''
        ARGUMENTS:
            k               logistic sigmoid constant (defaults to 1)
        '''
        self.k = k
        
    def clone(self):
        '''Return a new neuron with the same parameters.'''
        return Neuron(self.k)
    
    def input(self, a):
        '''Sets neuron input and calls activation function to set output.'''
        self.a = a
        self.y = self.g(a)
        return self.y
    
    def g(self, a):
        '''Neuron activation function'''
        from math import exp
        return 1. / (1. + exp(- self.k * a))
    
    def dy_da(self):
        '''Derivative of the activation function at the current activation level.'''
        return self.k * self.y * (1.0 - self.y)


class PerceptronLayer:
    '''A layer in a perceptron network.'''
    def __init__(self, numInputs, numNeurons, k=1.0, weights=None):
        '''
        ARGUMENTS:
            numInputs           Number of input features that will be passed to
                                the Perceptron for training and classification.
            numNeurons          Number of neurons in the Perceptron, which is
                                also the number of outputs.
        OPTIONAL ARGUMENTS:
            neuron              A prototype neuron that will be cloned to
                                populate the network.
            weights             initial weights with which to begin training
            rate                rate adjustment parameter.
        '''
        import random

	self.k = k
        numInputs += 1
        #if neuron:
        #    self.neuron = neuron
        #else:
        #    self.neuron = Neuron()
        if weights:
            if weights.shape != (numNeurons, numInputs):
                raise 'Shape of weight matrix does not match Perceptron shape.'
            self.weights = weights
        else:
            self.weights = numpy.array([1 - 2 * random.random() for i in range((numInputs) * numNeurons)])
            self.weights.shape = (numNeurons, numInputs)
        self.shape = [numNeurons, numInputs]
	self.x = numpy.empty(numInputs, float)

    def input(self, x):
        '''
        Sets Perceptron input, activates neurons and sets & returns Perceptron output.
        The Perceptron unity bias input (if used) should not be included in x.

        For classifying samples, call classify instead.
        '''
#        self.x = numpy.concatenate(([1.0], x))
        self.x[1:] = x
        self.z = numpy.dot(self.weights, self.x)
	self.y = self.g(self.z)
        return self.y

    def g(self, a):
        '''Neuron activation function'''
        return 1. / (1. + numpy.exp(- self.k * a))
    
    def dy_da(self):
        '''Derivative of the activation function at the current activation level.'''
        return self.k * (self.y * (1.0 - self.y))

class Perceptron:
    ''' A Multi-Layer Perceptron network with backpropagation learning.'''
    def __init__(self, layers, rate = 0.3, k=1.0):
        '''
        Creates the Perceptron network.

        ARGUMENTS:
            layers              A tuple specifying the network structure.
                                layers[0] is the number of inputs.
                                layers[-1] is the number of outputs
                                layers[1: -1] are # of units in each hidden layer.
        OPTIONAL ARGUMENTS:
            rate                Learning rate coefficient for weight adjustments
            neuron              A prototype neuron that is cloned to populate the network
        '''
        
        if type(layers) != list or len(layers) < 2:
            raise 'ERROR: Perceptron argument must be list of 2 or more integers'
        
        self.shape = layers[:]
        self.layers = [PerceptronLayer(layers[i - 1], layers[i], k) for i in range(1, len(layers))]

        self.rate = rate
        self.momentum = 0.8
        self.onIteration = None
        self.inputScale = 1
        self.stochastic = False
        
        self.trainingAccuracy = 0
        self.error = 0
        
        self._haveWeights = False

    def input(self, x):
        '''
        Sets Perceptron input, activates neurons and sets & returns Perceptron output.
        For classifying samples, call classify instead of input.
        '''
        self.x = x[:]
	
        x *= self.inputScale
        for layer in self.layers:
            x = layer.input(x)
        self.y = numpy.array(x)
        return x

    def classify(self, x):
        '''
        Classifies the given sample.  This has the same result as
        calling input and rounding the result.
        '''
        return [int(round(xx)) for xx in self.input(x)]

    def train(self, samples, max_iterations = 1000, accuracy = 100.0):
        '''
        Trains the Perceptron to classify the given samples.

        ARGUMENTS:
            samples             A list containing input-classification pairs. The
                                first element of each pair is the sample value.
                                The second element of each pair is the correct
                                classification of the sample, which should be a
                                list containing only 1's and 0's, with length
                                corresponding to the number of neurons in the
                                Perceptron.
            max_iterations      The maximum number of iterations to perform before
                                terminating training.
            accuracy            The accuracy at which to terminate training (if
                                max_iterations isn't reached first).
        '''
        from numpy import array, dot, transpose, zeros, repeat
        from spectral import status

        try:
                
            if not self._haveWeights:
                self.init_weights(samples)
                self._haveWeights = True
    
            for layer in self.layers:
                layer.dW_old = 0
            
            for iteration in xrange(max_iterations):
                
                self.reset_corrections()
                self.error = 0
                numSamples = 0
                numCorrect = 0
                self._sampleCount = 0
                self._iteration = iteration
    
                for (x, t) in samples:
    
                    numSamples += 1
                    self._sampleCount += 1
                    correct = self.classify(x) == t
                    if correct:
                        numCorrect += 1
                    delta = array(t) - self.y
                    self.error += sum(0.5 * delta * delta)
                    
                    # Determine incremental weight adjustments                
                    self.update_dWs(t)
                    if self.stochastic:
                        self.adjust_weights()
    
                self.trainingAccuracy = 100. * numCorrect / numSamples

                if self.onIteration and not self.onIteration(self):
                    return
                
                status.write('Iter % 5d: Accuracy = %.2f%% E = %f\n' % (iteration, self.trainingAccuracy, self.error))
                if self.trainingAccuracy >= accuracy:
                    status.write('Network trained to %.1f%% sample accuracy in %d iterations.\n' % (self.trainingAccuracy, iteration))
                    return
                
                if not self.stochastic:
                    self.adjust_weights()
                    
        except KeyboardInterrupt:
            print "KeyboardInterrupt: Terminating training."
            self.reset_corrections()
            return
            
        status.write('Terminating network training after %d iterations.\n' % iteration)

    def update_dWs(self, t):
        '''Update weight adjustment values for the current sample.'''
        from numpy import array, dot, zeros, zeros_like, transpose, newaxis
        
        # Output layer
        layerK = self.layers[-1]
#        dy_da = array([neuron.dy_da() for neuron in layerK.neurons], ndmin = 2)
        dy_da = array(layerK.dy_da(), ndmin = 2)
        dE_dy = t - self.y
        dE_dy.shape = (1, len(dE_dy))
        layerK.delta = dy_da * dE_dy
        dz_dW = array(layerK.x, ndmin=2)
        dW = dot(layerK.delta.transpose(), dz_dW)
        layerK.dW = layerK.dW + dW

        # Hidden layers
        for i in range(len(self.layers) - 2, -1, -1):
            (layerJ, layerK) = self.layers[i: i + 2]
            (J, K) = (layerJ.shape[0], layerK.shape[0])
            dW = zeros_like(layerJ.weights)
            layerJ.delta = zeros([1, J])
	    dy_da = layerJ.dy_da()
            for j in range(J):
                b = 0.0
                for k in range(K):
                    b += layerK.delta[0, k] * layerK.weights[k, j + 1]
                layerJ.delta[0, j] = dy_da[j] * b
                for c in range(layerJ.weights.shape[1]):
                    dW[j, c] = layerJ.delta[0, j] * layerJ.x[c]
            layerJ.dW = layerJ.dW + dW
        
    def adjust_weights(self):
        weights = [numpy.array(layer.weights) for layer in self.layers]
        try:
            for layer in self.layers:
                layer.dW = (self.rate / self._sampleCount) * layer.dW + self.momentum * layer.dW_old
                layer.weights += layer.dW
                layer.dW_old = numpy.array(layer.dW)
        except KeyboardInterrupt:
            print "Interrupt during weight adjustment. Restoring previous weights."
            for i in range(len(weights)):
                self.layers[i].weights = weights[i]
            raise
        finally:
            self.reset_corrections()
                    
    def reset_corrections(self):
        for layer in self.layers:
            layer.dW = numpy.zeros_like(layer.weights)
            
    def init_weights(self, samples):
        from random import random
        minMax = [(x, x) for x in samples[0][0]]
        for sample in samples[1:]:
            minMax = [(min(x[0], x[2]), max(x[1], x[2])) for x in zip(*zip(*minMax) + [sample[0]])]
        for i in range(len(self.shape) - 1):
            N = self.shape[i]
            if i > 0:
                minMax = [(-1, 1)] * N
            for j in range(self.shape[i + 1]):
                loc = [p[0] + random() * (p[1] - p[0]) for p in minMax]
                vec = numpy.array([random() - 0.5 for k in range(N + 1)])
                vec /= numpy.sqrt(sum(vec[1:]**2))
                vec[0] = numpy.sum(loc * vec[1:])
                self.layers[i].weights[j, :] = vec
        self._haveWeights = True
            
from spectral.algorithms.classifiers import SupervisedClassifier

class PerceptronSampleIterator:
    '''
    An iterator over all samples of all classes in a TrainingData object.
    Similar to Algorithms.SampleIterator but this on packages the samples
    to be used by the Perceptron.train method.

    For testing algoritms, the class variable max_samples can be set to an
    integer to specify the max number of samples to return for any
    training class.
    '''
    max_samples = None
    
    def __init__(self, trainingData):
        self.classes = trainingData
        
    def __iter__(self):
        i = 0
        ci = 0
        for cl in self.classes:
            t = [0] * len(self.classes)
            t[i] = 1
            j = 0
            for sample in cl:
                j += 1
                if self.max_samples and j > self.max_samples:
                    break
                yield (sample, t)
            i += 1

class PerceptronClassifier(Perceptron, SupervisedClassifier):
    def train(self, trainingClassData, max_iterations = 1000, accuracy = 100.0):
        '''
        Trains the Perceptron on the training data.
	
	Arguments:
	
	    `trainingClassData` (:class:`~spectral.algorithms.TrainingClassSet`):

		Data for the training classes.
	    
	    `max_iterations` (int):
	    
		Maximum number of training iterations to perform.
	    
	    `accuracy` (float):
	    
		Training set classification accuracy to which the classifier
		should be trained.
        '''
        # Number of Perceptron inputs must equal number of features in the
        # training data.
        if len(trainingClassData) != self.layers[-1].shape[0]:
            raise Exception('Number of nodes in output layer must match number of training classes.')
        self.trainingClassData = trainingClassData
        
        # Map output nodes to class indices
        self.indices = [cl.index for cl in self.trainingClassData]

        self.initialize_weights(trainingClassData)
        
        it = PerceptronSampleIterator(trainingClassData)
        Perceptron.train(self, it, max_iterations, accuracy)
        
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
        y = self.input(x)
        maxNodeIndex = numpy.argmax(y)
        val = int(round(y[maxNodeIndex]))
        # If val is zero, then no node was above threshold
        if val == 0:
            return 0
        else:
            return self.indices[maxNodeIndex]

    def initialize_weights(self, trainingClassData):
        '''
        Randomizes initial values of hidden layer weights and scale them to
        prevent overflow when evaluating activation function.

	Arguments:
	
	    `trainingClassData` (:class:`~spectral.algorithms.TrainingClassSet`):

		Data for the training classes.
        '''
        from spectral.algorithms.algorithms import SampleIterator
        from random import random

        maxVal = 0
        for sample in SampleIterator(trainingClassData):
            maxVal = max(max(numpy.absolute(sample.ravel())), maxVal)

        layer = self.layers[-2]
        for i in range(layer.shape[0]):
            layer.weights[i,0] = (random() * 2 - 1)
        self.inputScale = 1.0 / maxVal
	self._haveWeights = True


# Sample data

t2x1 = [
    [[0.1, 0.0], [1]],
    [[0.2, 0.1], [1]],
    [[0.3, 0.2], [1]],
    [[0.4, 0.3], [1]],
    [[0.5, 0.4], [1]],
    [[0.6, 0.5], [1]],
    [[0.7, 0.6], [1]],
    [[0.8, 0.7], [1]],
    [[0.9, 0.8], [1]],
    [[1.0, 0.9], [1]],
    [[0.0, 0.1], [0]],
    [[0.1, 0.2], [0]],
    [[0.2, 0.3], [0]],
    [[0.3, 0.4], [0]],
    [[0.4, 0.5], [0]],
    [[0.5, 0.6], [0]],
    [[0.6, 0.7], [0]],
    [[0.7, 0.8], [0]],
    [[0.8, 0.9], [0]],
    [[0.9, 1.0], [0]]
    ]

t2x2 = [
    [[5, 2], [1, 0]],
    [[3, 1], [1, 0]],
    [[8, 1], [1, 0]],
    [[5, 3], [1, 0]],
    [[2, 0], [1, 0]],
    [[0, 4], [1, 0]],
    [[2, 4], [1, 0]],
    [[1, 7], [1, 0]],
    [[3, 5], [1, 0]],
    [[2, 6], [1, 0]],
    [[4, 9], [0, 1]],
    [[2, 9], [0, 1]],
    [[4, 9], [0, 1]],
    [[6, 8], [0, 1]],
    [[5, 7], [0, 1]],
    [[9, 3], [0, 1]],
    [[7, 4], [0, 1]],
    [[8, 5], [0, 1]],
    [[7, 6], [0, 1]],
    [[9, 8], [0, 1]]
    ]

t4 = [
    [[5, 2], [1, 0, 0, 0]],
    [[3, 1], [1, 0, 0, 0]],
    [[8, 1], [1, 0, 0, 0]],
    [[5, 3], [1, 0, 0, 0]],
    [[2, 0], [1, 0, 0, 0]],
    [[0, 4], [0, 1, 0, 0]],
    [[2, 4], [0, 1, 0, 0]],
    [[1, 7], [0, 1, 0, 0]],
    [[3, 5], [0, 1, 0, 0]],
    [[2, 6], [0, 1, 0, 0]],
    [[4, 9], [0, 0, 1, 0]],
    [[2, 9], [0, 0, 1, 0]],
    [[4, 9], [0, 0, 1, 0]],
    [[6, 8], [0, 0, 1, 0]],
    [[5, 7], [0, 0, 1, 0]],
    [[9, 3], [0, 0, 0, 1]],
    [[7, 4], [0, 0, 0, 1]],
    [[8, 5], [0, 0, 0, 1]],
    [[7, 6], [0, 0, 0, 1]],
    [[9, 8], [0, 0, 0, 1]]
    ]

xor_data = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]],
]

t2x1b = [[a, [b[1]]] for [a, b] in t2x2]

def go():
    #p = Perceptron([2, 1])
    #p.train(t2x1, 5000)
    #
    
    #p = Perceptron([2, 1, 2])
    #p.momentum = 0.5
    #import random
    #p.layers[0].weights[0,0] = random.random() * 20 - 10
    #p.train(t2x2, 1000)
    
    #
    #p = Perceptron([2, 4, 4])
    #p.stochastic = True
    #p.train(t4, 5000)
    #
    #p = Perceptron([2, 4, 4, 4])
    #p.stochastic = True
    #p.train(t4, 5000)
    
    p = Perceptron([2, 1])
    p.train(t2x1b, 5000)

    return p

if __name__ == '__main__':
    p = go()
