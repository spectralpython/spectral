#########################################################################
#
#   perceptron.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2001-2014 Thomas Boggs
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

import numpy as np
import sys

class PerceptronLayer:
    '''A multilayer perceptron layer with sigmoid activation function.'''
    def __init__(self, shape, k=1.0, weights=None):
        '''
        Arguments:

            `shape` (2-tuple of int):

                Should have the form (`num_inputs`, `num_neurons`), where
                `num_inputs` does not include an input for the bias weights.

            `k` (float):

                Sigmoid shape parameter.

            `weights` (ndarray):

                Initial weights for the layer. Note that if provided, this
                argument must have shape (`num_neurons`, `num_inputs` + 1). If
                not provided, initial weights will be randomized.
        '''
        self.k = k
        self.shape = (shape[1], shape[0] + 1)
        if weights:
            if weights.shape != self.shape:
                raise Exception('Shape of weight matrix does not ' \
                                'match Perceptron layer shape.')
            self.weights = np.array(weights, dtype=np.float64)
        else:
            self.randomize_weights()
        self.dW = np.zeros_like(self.weights)
        self.dW_buf = np.zeros_like(self.dW)
        self.x = np.ones(self.shape[1], float)

    def randomize_weights(self):
        '''Randomizes the layer weight matrix.
        The bias weight will be in the range [0, 1). The remaining weights will
        correspond to a vector with unit length and uniform random orienation.
        '''
        import math
        self.weights = 1. - 2. * np.random.rand(*self.shape)
        for row in self.weights:
            row[1:] /= math.sqrt(np.sum(row[1:]**2))
#            row[0] = 0.5 - np.random.rand() - 0.5 * np.sum(row[1:])
            row[0] = -0.5 * np.random.rand() - 0.5 * np.sum(row[1:])

    def input(self, x, clip=0.0):
        '''Sets layer input and computes output.

        Arguments:

            `x` (sequence):

                Layer input, not including bias input.

            `clip` (float >= 0):

                Optional clipping value to limit sigmoid output. The sigmoid
                function has output in the range (0, 1). If the `clip` argument
                is set to `a` then all neuron outputs for the layer will be
                constrained to the range [a, 1 - a]. This can improve perceptron
                learning rate in some situations.

        Return value:

            The ndarray of output values is returned and is also set in the `y`
            attribute of the layer.

        For classifying samples, call `classify` instead.
        '''
        self.x[1:] = x
        self.z = np.dot(self.weights, self.x)
        if clip > 0.:
            self.y = np.clip(self.g(self.z), clip, 1. - clip)
        else:
            self.y = self.g(self.z)
        return self.y

    def g(self, a):
        '''Neuron activation function (logistic sigmoid)'''
        return 1. / (1. + np.exp(- self.k * a))

    def dy_da(self):
        '''Derivative of activation function at current activation level.'''
        return self.k * (self.y * (1.0 - self.y))


class Perceptron:
    ''' A Multi-Layer Perceptron network with backpropagation learning.'''
    def __init__(self, layers, k=1.0):
        '''
        Creates the Perceptron network.

        Arguments:

            layers (sequence of integers):

                A specifying the network structure. `layers`[0] is the number
                of inputs. `layers`[-1] is the number of perceptron outputs.
                `layers`[1: -1] are the numbers of units in the hidden layers.

            `rate` (float):

                Learning rate coefficient for weight adjustments

            `k` (float):

                Sigmoid shape parameter.
        '''
        if type(layers) != list or len(layers) < 2:
            raise Exception('ERROR: Perceptron argument must be list of 2 or '
                            'more integers.')
        self.shape = layers[:]
        self.layers = [PerceptronLayer((layers[i - 1], layers[i]), k)
                       for i in range(1, len(layers))]
        self.accuracy = 0
        self.error = 0

        # To prevent overflow when scaling inputs
        self.min_input_diff = 1.e-8

        self.cache_weights = True


    def input(self, x, clip=0.0):
        '''Sets Perceptron input, activates neurons and sets & returns output.

        Arguments:

            `x` (sequence):

                Inputs to input layer. Should not include a bias input.


            `clip` (float >= 0):

                Optional clipping value to limit sigmoid output. The sigmoid
                function has output in the range (0, 1). If the `clip` argument
                is set to `a` then all neuron outputs for the layer will be
                constrained to the range [a, 1 - a]. This can improve perceptron
                learning rate in some situations.

        For classifying samples, call `classify` instead of `input`.
        '''
        self.x = x[:]
        x = self._scale * (x - self._offset)
        for layer in self.layers:
            x = layer.input(x, clip)
        self.y = np.array(x)
        return x

    def classify(self, x):
        '''Classifies the given sample.
        This has the same result as calling input and rounding the result.
        '''
        return [int(round(xx)) for xx in self.input(x)]

    def train(self, X, Y, max_iterations=1000, accuracy=100.0, rate=0.3,
              momentum=0.1, batch=0, clip=0.0, on_iteration=None,
              status=sys.stdout):
        '''
        Trains the Perceptron to classify the given samples.

        Arguments:

            `X`:

                The sequence of observations to be learned. Each element of `X`
                must have a length corresponding to the input layer of the
                network. Values in `X` are not required to be scaled.

            `Y`:

                Truth values corresponding to elements of `X`. `Y` must contain
                as many elements as `X` and each element of `Y` must contain a
                number of elements corresponding to the output layer of the
                network. All values in `Y` should be in the range [0, 1] and for
                training a classifier, values in `Y` are typically *only* 0 or 1
                (i.e., no intermediate values).

            `max_iterations` (int):

                Maximum number of iterations through the data to perform.
                Training will end sooner if the specified accuracy is reached in
                fewer iterations.

            `accuracy` (float):

                The percent training accuracy at which to terminate training, if
                the maximum number of iterations are not reached first. This
                value can be set greater than 100 to force a specified number of
                training iterations to be performed (e.g., to continue reducing
                the error term after 100% classification accuracy has been
                achieved.

            `rate` (float):

                The perceptron learning rate (typically in the range (0, 1]).

            `momentum` (float):

                The perceptron learning momentum term, which specifies the
                fraction of the previous update value that should be added to
                the current update term. The value should be in the range [0, 1).

            `batch` (positive integer):

                Specifies how many samples should be evaluated before an update
                is made to the perceptron weights. A value of 0 indicates batch
                updates should be performed (evaluate all training inputs prior
                to updating). Otherwise, updates will be aggregated for every
                `batch` inputs (i.e., `batch` == 1 is stochastic learning).

            `clip` (float >= 0):

                Optional clipping value to limit sigmoid output during training.
                The sigmoid function has output in the range (0, 1). If the
                `clip` argument is set to `a` then all neuron outputs for the
                layer will be constrained to the range [a, 1 - a]. This can
                improve perceptron learning rate in some situations.

                After training the perceptron with a clipping value, `train` can
                be called again with clipping set to 0 to continue reducing the
                training error.

            `on_iteration` (callable):

                A callable object that accepts the perceptron as input and
                returns bool. If this argument is set, the object will be called
                at the end of each training iteration with the perceptron as its
                argument. If the callable returns True, training will terminate.

            `status`:

                An object with a `write` method that can be set to redirect
                training status messages somewhere other than stdout. To
                suppress output, set `stats` to None.
        '''
        import itertools
        import os

        if status is None:
            status = open(os.devnull, 'w')

        try:
            self._set_scaling(X)
            for layer in self.layers:
                layer.dW_old = np.zeros_like(layer.dW)

            for iteration in xrange(max_iterations):

                self._reset_corrections()
                self.error = 0
                num_samples = 0
                num_correct = 0
                num_summed = 0

                for (x, t) in itertools.izip(X, Y):
                    num_samples += 1
                    num_summed += 1
                    num_correct += np.all(np.round(self.input(x, clip)) == t)
                    delta = np.array(t) - self.y
                    self.error += 0.5 * sum(delta**2)

                    # Determine incremental weight adjustments
                    self._update_dWs(t)
                    if batch > 0 and num_summed == batch:
                        self._adjust_weights(rate, momentum, num_summed)
                        num_summed = 0

                # In case a partial batch is remaining
                if batch > 0 and num_summed > 0:
                    self._adjust_weights(rate, momentum, num_summed)
                    num_summed = 0

                self.accuracy = 100. * num_correct / num_samples

                if on_iteration and on_iteration(self):
                    return True

                status.write('Iter % 5d: Accuracy = %.2f%% E = %f\n' %
                             (iteration, self.accuracy, self.error))
                if self.accuracy >= accuracy:
                    status.write('Network trained to %.1f%% sample accuracy '
                                 'in %d iterations.\n'
                                 % (self.accuracy, iteration))
                    return True

                # If doing full batch learning (batch == 0)
                if num_summed > 0:
                    self._adjust_weights(rate, momentum, num_summed)
                    num_summed = 0

        except KeyboardInterrupt:
            status.write("KeyboardInterrupt: Terminating training.\n")
            self._reset_corrections()
            return False

        status.write('Terminating network training after %d iterations.\n' %
                     iteration)
        return False

    def _update_dWs(self, t):
        '''Update weight adjustment values for the current sample.'''

        # Output layer:
        #   dE/dy = t - y
        #   dz/dW = x
        layerK = self.layers[-1]
        layerK.delta = layerK.dy_da() * (t - self.y)
        layerK.dW += np.outer(layerK.delta, layerK.x)

        # Hidden layers
        for i in range(len(self.layers) - 2, -1, -1):
            (layerJ, layerK) = self.layers[i: i + 2]
            b = np.dot(layerK.delta, layerK.weights[:, 1:])
            layerJ.delta = layerJ.dy_da() * b
            layerJ.dW += np.outer(layerJ.delta, layerJ.x)

    def _adjust_weights(self, rate, momentum, num_summed):
        '''Applies aggregated weight adjustments to the perceptron weights.'''
        if self.cache_weights:
            weights = [np.array(layer.weights) for layer in self.layers]
        try:
            if momentum > 0:
                for layer in self.layers:
                    layer.dW *= rate
#                    layer.dW *= (rate / num_summed)
                    layer.dW += momentum * layer.dW_old
                    layer.weights += layer.dW
                    (layer.dW_old, layer.dW) = (layer.dW, layer.dW_old)
            else:
                for layer in self.layers:
                    layer.dW *= rate
 #                   layer.dW *= (rate / num_summed)
                    layer.weights += layer.dW
        except KeyboardInterrupt:
            if self.cache_weights:
                status.write('Interrupt during weight adjustment. Restoring ' \
                            'previous weights.\n')
                for i in range(len(weights)):
                    self.layers[i].weights = weights[i]
            else:
                status.write('Interrupt during weight adjustment. Weight ' \
                            'cacheing was disabled so current weights may' \
                            'be corrupt.\n')
            raise
        finally:
            self._reset_corrections()

    def _adjust_weights_working(self, rate, momentum, num_summed):
        '''Applies aggregated weight adjustments to the perceptron weights.'''
        if self.cache_weights:
            weights = [np.array(layer.weights) for layer in self.layers]
        try:
            for layer in self.layers:
                layer.dW *= (rate / num_summed)
                if momentum > 0:
                    layer.dW += momentum * layer.dW_old
#                layer.dW = (rate / num_summed) * \
#                    layer.dW + momentum * layer.dW_old
                layer.weights += layer.dW
                (layer.dW_old, layer.dW) = (layer.dW, layer.dW_old)
        except KeyboardInterrupt:
            if self.cache_weights:
                status.write('Interrupt during weight adjustment. Restoring ' \
                            'previous weights.\n')
                for i in range(len(weights)):
                    self.layers[i].weights = weights[i]
            else:
                status.write('Interrupt during weight adjustment. Weight ' \
                            'cacheing was disabled so current weights may' \
                            'be corrupt.\n')
            raise
        finally:
            self._reset_corrections()

    def _reset_corrections(self):
        for layer in self.layers:
            layer.dW.fill(0)

    def _set_scaling(self, X):
        mins = maxes = None
        for x in X:
            if mins is None:
                mins = x
                maxes = x
            else:
                mins = np.min([mins, x], axis=0)
                maxes = np.max([maxes, x], axis = 0)
        self._offset = mins
        r = maxes - mins
        self._scale = 1. / np.where(r < self.min_input_diff, 1, r)
        

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

and_data = [
    [[0, 0], [0]],
    [[0, 1], [0]],
    [[1, 0], [0]],
    [[1, 1], [1]],
]

xor_data1 = [
    [[-1, -1], [0]],
    [[-1,  6], [1]],
    [[ 6, -1], [1]],
    [[ 6,  6], [0]],
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

def run_tests():
    (X, Y) = zip(*xor_data)
    p = Perceptron([2, 3, 1])
    r1 = p.train(X, Y, 20000, rate=0.7, momentum=0.2, batch=1, clip=0.)

    (X, Y) = zip(*and_data)
    p = Perceptron([2, 1])
    r2 = p.train(X, Y, 20000, rate=0.7, momentum=0.2, batch=1, clip=0.)

    (X, Y) = zip(*t4)
    p = Perceptron([2, 4, 4, 4])
    r3 = p.train(X, Y, 20000, rate=0.7, momentum=0.2, batch=1, clip=0.)

    if r1:
        print "XOR test passed."
    else:
        print "XOR test FAILED"
    if r2:
        print "AND test passed."
    else:
        print "AND test FAILED"
    if r3:
        print "4-class test passed."
    else:
        print "4-class test FAILED"

if __name__ == '__main__':
    from spectral.algorithms.perceptron import run_tests
    run_tests()
