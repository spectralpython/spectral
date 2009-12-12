#########################################################################
#
#   Spectral.py - This file is part of the Spectral Python (SPy) package.
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
    
    def input(self, x):
        '''Sets neuron input and calls activation function to set output.'''
        self.z = x
        self.y = self.g(x)
        return self.y
    
    def g(self, x):
        '''Neuron activation function'''
        from math import exp
        return 1. / (1. + exp(-x / self.k))
    
    def dg_dz(self):
        '''Derivative of the activation function at the current activation level.'''
        return self.y * (1.0 - self.y) / self.k

class Perceptron:
    '''
    A single-layer neural network with back-propagation learning.
    '''
    def __init__(self, numInputs, numNeurons, bias = True, neuron = None, weights = None,rate = 0.1):
        '''
        Creates the Perceptron network.

        ARGUMENTS:
            numInputs           Number of input features that will be passed to
                                the Perceptron for training and classification.
            numNeurons          Number of neurons in the Perceptron, which is
                                also the number of outputs.
        OPTIONAL ARGUMENTS:
            bias                boolean value indicating whether to use a unity
                                bias input to the network. Default is True.
            neuron              A prototype neuron that will be cloned to
                                populate the network.
            weights             initial weights with which to begin training
            rate                initial rate adjustment parameter.
        '''
        import random

        self.rate = rate
        self.bias = bias
        if bias:
            numInputs += 1
        if neuron:
            self.neuron = neuron
        else:
            self.neuron = Neuron()
        if weights:
            if weights.shape != (numNeurons, numInputs):
                raise 'Shape of weight matrix does not match Perceptron shape.'
            self.weights = weights
        else:
            self.weights = numpy.array([1 - 2 * random.random() for i in range((numInputs) * numNeurons)])
            self.weights.shape = (numNeurons, numInputs)
        self.shape = [numNeurons, numInputs]
        self.neurons = [self.neuron.clone() for i in range(numNeurons)]

        self.rateInc = 1.1      # multiplier for increasing descent rate
        self.rateDec = 0.5      # multiplier for decreasing descent rate

    def input(self, x):
        '''
        Sets Perceptron input, activates neurons and sets & returns Perceptron output.
        The Perceptron unity bias input (if used) should not be included in x.

        For classifying samples, call classify instead of input.
        '''
        if self.bias:
            self.x = numpy.concatenate(([1.0], x))
        else:
            self.x = x
        self.z = numpy.dot(self.weights, self.x)
        self.y = numpy.array([neuron.input(a) for (neuron, a) in zip(self.neurons, self.z)])
        return self.y

    def classify(self, x):
        '''
        Classifies the given sample.  This has the same result as
        calling input and rounding the result.
        '''
        return map(round, self.input(x))

    def train(self, samples, maxIterations = 10000, accuracy = 100.):
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
            maxIterations       The maximum number of iterations to perform before
                                terminating training.
            accuracy            The accuracy at which to terminate training (if
                                maxIterations isn't reached first).
        '''
        from numpy import array, zeros, dot, transpose, repeat
        from Spectral import status
        
        numSamples = len(samples)
        self.oldE = 1.e200
        
        for iteration in range(1, maxIterations + 1):

            numCorrect = 0
          
            self.dW = zeros(self.weights.shape, float)
            E = 0

            for (x, t) in samples:

                correct = self.classify(x) == t
                if correct:
                    numCorrect += 1

                # Determine incremental weight adjustment

                t = array(t)
                delta = t - self.y
                error = 0.5 * delta * delta
                E += sum(error)
                
                K = self.shape[0]
                y = array(self.y)
                dy_dz = array([neuron.dg_dz() for neuron in self.neurons])
                dy_dz.shape = (1, len(dy_dz))
                dE_dy = -(t - y)
                dE_dy.shape = (1, len(dE_dy))
                dz_dW = array(self.x)
                dz_dW.shape = (1, len(dz_dW))
                repeat(dz_dW, K)
                dW = - self.rate * dot(transpose(dy_dz * dE_dy), dz_dW)
                self.dW = self.dW + dW

            # If E increased, then the previous adjustment was too big.

            if E > self.oldE:
                self.weights = self.oldWeights
                self.rate *= self.rateDec
                status.write('Iter % 5d:\n' % iteration)
            else:
                self.oldWeights = self.weights
                self.oldE = E
                self.weights = self.weights + self.dW / float(len(samples))
                self.rate *= self.rateInc
                curAccuracy = 100.0 * numCorrect / numSamples                  
                status.write('Iter % 5d: Accuracy = %.2f%% E = %f, rate = %f\n' % (iteration, curAccuracy, E, self.rate))
                if curAccuracy >= accuracy:
                    status.write('Network trained in %d iterations.\n' % iteration)
                    return
        status.write('Terminating network training after %d iterations.\n' % iteration)

class MultiLayerPerceptron:
    ''' A Multi-Layer Perceptron classifier with a single hidden layer.'''
    def __init__(self, layers, neuron = None, rate = 0.1):
        '''
        Creates the Perceptron network.

        ARGUMENTS:
            layers              A 3-tuple whose elements specify the number of
                                network inputs, number of hidden-layer neurons
                                and number of outputs, respectively.
        OPTIONAL ARGUMENTS:
            bias                boolean value indicating whether to include a
                                bias input to the network.
            neuron              A prototype neuron that will be cloned to
                                populate the network
            weights             initial weights with which to begin training
            rate                initial rate adjustment parameter.
        '''
        
        if type(layers) != list or len(layers) != 3:
            raise 'Expecting 3-tuple for Perceptron layer sizes.'

        self.inputSize = layers[0]
        # Use bias inputs on all but the output layer
        biases = [True] * (len(layers) - 2) + [False]
        self.layers = [Perceptron(layers[i - 1], layers[i], neuron = neuron, bias = biases[i - 1]) for i in range(1, len(layers))]

        self.rate = rate
        self.rateInc = 1.1
        self.rateDec = 0.5
        self.inputScale = 1

    def input(self, x):
        '''
        Sets Perceptron input, activates neurons and sets & returns Perceptron output.

        For classifying samples, call classify instead of input.
        '''
        self.x = x
        x = [self.inputScale * xx for xx in x]
        for layer in self.layers:
            x = layer.input(x)
        self.y = numpy.array(x)
        return x

    def classify(self, x):
        '''
        Classifies the given sample.  This has the same result as
        calling input and rounding the result.
        '''
        return map(round, self.input(x))

    def train(self, samples, maxIterations = 1000, accuracy = 100.0):
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
            maxIterations       The maximum number of iterations to perform before
                                terminating training.
            accuracy            The accuracy at which to terminate training (if
                                maxIterations isn't reached first).
        '''
        from numpy import array, dot, transpose, zeros, repeat
        from Spectral import status

        self.oldE = 1.e200
        self.rateCount = 0
        for layer in self.layers:
            layer.oldWeights = layer.weights
        
        for iteration in range(maxIterations):
            
            self.resetCorrections()
            E = 0
            numSamples = 0
            numCorrect = 0

            for (x, t) in samples:

                numSamples += 1

                correct = self.classify(x) == t
                if correct:
                    numCorrect += 1

                t = array(t)
                delta = t - self.y
                error = 0.5 * delta * delta
                E += sum(error)
                
                # Determine incremental weight adjustments
                
                # Output layer
                layerK = self.layers[-1]
                K = layerK.shape[0]
                layerJ = self.layers[-2]
                J = layerJ.shape[0]
                y = array(layerK.y)
                dy_dz = array([neuron.dg_dz() for neuron in layerK.neurons])
                dy_dz.shape = (1, len(dy_dz))
                dE_dy = -(t - y)
                dE_dy.shape = (1, len(dE_dy))
                dz_dW = array(layerK.x)
                dz_dW.shape = (1, len(dz_dW))
                repeat(dz_dW, K)
                dW = - self.rate * dot(transpose(dy_dz * dE_dy), dz_dW)
                layerK.dW = layerK.dW + dW

                # Hidden layer
                I = self.inputSize
                dW = zeros(layerJ.weights.shape, float)
                for j in range(J):
                    b = 0.0
                    for k in range(K):
                        b += delta[k] * layerK.neurons[k].dg_dz() * layerK.weights[k, j]
                    a = self.rate * layerJ.neurons[j].dg_dz()
                    for i in range(layerJ.weights.shape[1]):
                        dW[j, i] = a * layerJ.x[i] * b
                layerJ.dW = layerJ.dW + dW

            if E > self.oldE:
                # Error is increasing so decrease descent rate
                self.rate = self.rateDec * self.rate
                layerJ.weights = layerJ.oldWeights
                layerK.weights = layerK.oldWeights
                self.resetCorrections()
                status.write('Iter % 5d:\n' % iteration)
                continue
            else:
                # Adjust weights and increase descent rate
                curAccuracy = 100. * numCorrect / numSamples
                status.write('Iter % 5d: Accuracy = %.2f%% E = %f rate = %f\n' % (iteration, curAccuracy, E, self.rate))
                if curAccuracy >= accuracy:
                    status.write('Network trained to %.1f%% sample accuracy in %d iterations.\n' % (curAccuracy, iteration))
                    return
                self.oldE = E
                layerJ.oldWeights = layerJ.weights
                layerJ.weights = layerJ.weights + layerJ.dW / float(numSamples)
                layerK.oldWeights = layerK.weights
                layerK.weights = layerK.weights + layerK.dW / float(numSamples)
                self.rate = self.rateInc * self.rate

        status.write('Terminating network training after %d iterations.\n' % iteration)
                    
    def resetCorrections(self):
        for layer in self.layers:
            layer.dW = numpy.zeros(layer.weights.shape, numpy.float)


from Spectral.Algorithms.Classifiers import SupervisedClassifier

class PerceptronSampleIterator:
    '''
    An iterator over all samples of all classes in a TrainingData object.
    Similar to Algorithms.SampleIterator but this on packages the samples
    to be used by the Perceptron.train method.

    For testing algoritms, the class variable maxSamples can be set to an
    integer to specify the max number of samples to return for any
    training class.
    '''
    maxSamples = None
    
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
                if self.maxSamples and j > self.maxSamples:
                    break
                yield (sample, t)
            i += 1

class PerceptronClassifier(MultiLayerPerceptron, SupervisedClassifier):
    def train(self, trainingClassData, maxIterations = 1000, accuracy = 100.0):
        '''
        Trains the Perceptron to classify the training data.
        '''
        # Number of Perceptron inputs must equal number of features in the
        # training data.
        if len(trainingClassData) != self.layers[-1].shape[0]:
            raise 'Number of nodes in output layer must match number of training classes.'
        self.trainingClassData = trainingClassData
        
        # Map output nodes to class indices
        self.indices = [cl.index for cl in self.trainingClassData]

        self.initializeWeights(trainingClassData)
        
        it = PerceptronSampleIterator(trainingClassData)
        MultiLayerPerceptron.train(self, it, maxIterations, accuracy)
        
    def classifySpectrum(self, x):
        '''Determine in which class the sample belongs.'''
        y = self.input(x)
        maxNodeIndex = numpy.argmax(y)
        val = int(round(y[maxNodeIndex]))
        # If val is zero, then no node was above threshold
        if val == 0:
            return 0
        else:
            return self.indices[maxNodeIndex]

    def initializeWeights(self, trainingClassData):
        '''
        Randomize initial values of hidden layer weights and scale them to
        prevent overflow when evaluating activation function.
        '''
        from Spectral.Algorithms.Algorithms import SampleIterator
        from random import random

        maxVal = 0
        for sample in SampleIterator(trainingClassData):
            maxVal = max(max(numpy.absolute(sample.ravel())), maxVal)

        layer = self.layers[-2]
        for i in range(layer.shape[0]):
            layer.weights[i,0] = (random() * 2 - 1)
        self.inputScale = 1.0 / maxVal
        

if __name__ == '__main__':

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
    p = Perceptron(2, 1, rate = 0.1)
    p.train(t2x1)

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
    p = Perceptron(2, 2, rate = 0.1)
    p.train(t2x2, 1000)

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
    mlp = MultiLayerPerceptron([2, 4, 4], rate = 0.1)
    mlp.train(t4, 5000)

