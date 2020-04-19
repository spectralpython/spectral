'''
Classes and functions for classification with neural networks.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import itertools
import math
import numpy as np
import os
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
        self.weights = 1. - 2. * np.random.rand(*self.shape)
        for row in self.weights:
            row[1:] /= math.sqrt(np.sum(row[1:]**2))
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

                A list specifying the network structure. `layers`[0] is the number
                of inputs. `layers`[-1] is the number of perceptron outputs.
                `layers`[1: -1] are the numbers of units in the hidden layers.

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

        # If True, previous iteration weights are preserved after interrupting
        # training (with CTRL-C)
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

    def train(self, X, Y, max_iterations=10000, accuracy=100.0, rate=0.3,
              momentum=0., batch=1, clip=0.0, on_iteration=None,
              stdout=sys.stdout):
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

            `stdout`:

                An object with a `write` method that can be set to redirect
                training status messages somewhere other than stdout. To
                suppress output, set `stats` to None.
        '''
        if stdout is None:
            stdout = open(os.devnull, 'w')

        try:
            self._set_scaling(X)
            for layer in self.layers:
                layer.dW_old = np.zeros_like(layer.dW)

            for iteration in range(max_iterations):

                self._reset_corrections()
                self.error = 0
                num_samples = 0
                num_correct = 0
                num_summed = 0

                for (x, t) in zip(X, Y):
                    num_samples += 1
                    num_summed += 1
                    num_correct += np.all(np.round(self.input(x, clip)) == t)
                    delta = np.array(t) - self.y
                    self.error += 0.5 * sum(delta**2)

                    # Determine incremental weight adjustments
                    self._update_dWs(t)
                    if batch > 0 and num_summed == batch:
                        self._adjust_weights(rate, momentum, num_summed,
                                             stdout)
                        num_summed = 0

                # In case a partial batch is remaining
                if batch > 0 and num_summed > 0:
                    self._adjust_weights(rate, momentum, num_summed, stdout)
                    num_summed = 0

                self.accuracy = 100. * num_correct / num_samples

                if on_iteration and on_iteration(self):
                    return True

                stdout.write('Iter % 5d: Accuracy = %.2f%% E = %f\n' %
                             (iteration, self.accuracy, self.error))
                if self.accuracy >= accuracy:
                    stdout.write('Network trained to %.1f%% sample accuracy '
                                 'in %d iterations.\n'
                                 % (self.accuracy, iteration + 1))
                    return True

                # If doing full batch learning (batch == 0)
                if num_summed > 0:
                    self._adjust_weights(rate, momentum, num_summed, stdout)
                    num_summed = 0

        except KeyboardInterrupt:
            stdout.write("KeyboardInterrupt: Terminating training.\n")
            self._reset_corrections()
            return False

        stdout.write('Terminating network training after %d iterations.\n' %
                     (iteration + 1))
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

    def _adjust_weights(self, rate, momentum, num_summed, stdout):
        '''Applies aggregated weight adjustments to the perceptron weights.'''
        if self.cache_weights:
            weights = [np.array(layer.weights) for layer in self.layers]
        try:
            if momentum > 0:
                for layer in self.layers:
                    layer.dW *= (float(rate) / num_summed)
                    layer.dW += momentum * layer.dW_old
                    layer.weights += layer.dW
                    (layer.dW_old, layer.dW) = (layer.dW, layer.dW_old)
            else:
                for layer in self.layers:
                    layer.dW *= (float(rate) / num_summed)
                    layer.weights += layer.dW
        except KeyboardInterrupt:
            if self.cache_weights:
                stdout.write('Interrupt during weight adjustment. Restoring ' \
                            'previous weights.\n')
                for i in range(len(weights)):
                    self.layers[i].weights = weights[i]
            else:
                stdout.write('Interrupt during weight adjustment. Weight ' \
                            'cacheing was disabled so current weights may' \
                            'be corrupt.\n')
            raise
        finally:
            self._reset_corrections()

    def _reset_corrections(self):
        for layer in self.layers:
            layer.dW.fill(0)

    def _set_scaling(self, X):
        '''Sets translation/scaling of inputs to map X to the range [0, 1].'''
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

xor_data = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]],
]

xor_data2 = [
    [[0, 0], [0, 1]],
    [[0, 1], [1, 0]],
    [[1, 0], [1, 0]],
    [[1, 1], [0, 1]],
]

and_data = [
    [[0, 0], [0]],
    [[0, 1], [0]],
    [[1, 0], [0]],
    [[1, 1], [1]],
]

def test_case(XY, shape, *args, **kwargs):
    (X, Y) = list(zip(*XY))
    p = Perceptron(shape)
    trained = p.train(X, Y, *args, **kwargs)
    return (trained, p)
    
def test_xor(*args, **kwargs):
    XY = xor_data
    shape = [2, 2, 1]
    return test_case(XY, shape, *args, **kwargs)

def test_xor222(*args, **kwargs):
    XY = xor_data2
    shape = [2, 2, 2]
    return test_case(XY, shape, *args, **kwargs)

def test_xor231(*args, **kwargs):
    XY = xor_data
    shape = [2, 3, 1]
    return test_case(XY, shape, *args, **kwargs)

def test_and(*args, **kwargs):
    XY = and_data
    shape = [2, 1]
    return test_case(XY, shape, *args, **kwargs)

if __name__ == '__main__':
    tests = [('AND (2x1)', test_and),
             ('XOR (2x2x1)', test_xor),
             ('XOR (2x2x2)', test_xor222),
             ('XOR (2x3x1)', test_xor231)]
    results = [test[1](5000)[0] for test in tests]
    nr = [(p[0][0], p[1]) for p in zip(tests, results)]
    print()
    print('Training results for 5000 iterations')
    print('------------------------------------')
    for (name, result) in nr:
        s = [ 'FAILED', 'PASSED'][result]
        print('{0:<20}: {1}'.format(name, s))
    if False in results:
        print('\nNote: XOR convergence for these small network sizes is')
        print('dependent on initial weights, which are randomized. Try')
        print('running the test again.')
