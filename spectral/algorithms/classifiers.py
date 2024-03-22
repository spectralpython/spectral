'''
Supervised classifiers and base class for all classifiers.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import numpy as np

import spectral as spy
from .algorithms import GaussianStats, ImageIterator
from .detectors import RX
from .perceptron import Perceptron

__all__ = ('GaussianClassifier', 'MahalanobisDistanceClassifier',
           'PerceptronClassifier')


class Classifier(object):
    '''
    Base class for Classifiers.  Child classes must implement the
    classify_spectrum method.
    '''
    # It is often faster to compute the detector/classifier scores for the
    # entire image for each class, rather than for each class on a per-pixel
    # basis. However, this significantly increases memory requirements. If
    # the following parameter is True, class scores will be computed for the
    # entire image.
    cache_class_scores = True

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
        status = spy._status
        status.display_percentage('Classifying image...')
        it = ImageIterator(image)
        class_map = np.zeros(image.shape[:2], np.int16)
        N = it.get_num_elements()
        i, inc = (0, N / 100)
        for spectrum in it:
            class_map[it.row, it.col] = self.classify_spectrum(spectrum)
            i += 1
            if not i % inc:
                status.update_percentage(float(i) / N * 100.)
        status.end_percentage()
        return class_map

    def classify(self, X, **kwargs):
        if X.ndim == 1:
            return self.classify_spectrum(X, **kwargs)
        else:
            return self.classify_image(X, **kwargs)


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
        logger = logging.getLogger('spectral')
        if not self.min_samples:
            # Set minimum number of samples to the number of bands in the image
            self.min_samples = training_data.nbands
            logger.info('Setting min samples to %d', self.min_samples)
        self.classes = []
        for cl in training_data:
            if cl.size() >= self.min_samples:
                self.classes.append(cl)
            else:
                logger.warn('Omitting class %3d : only %d samples present',
                            cl.index, cl.size())
        for cl in self.classes:
            if not hasattr(cl, 'stats') or not cl.stats_valid():
                cl.calc_stats()

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
        scores = np.empty(len(self.classes))
        for (i, cl) in enumerate(self.classes):
            delta = (x - cl.stats.mean)
            scores[i] = math.log(cl.class_prob) - 0.5 * cl.stats.log_det_cov \
              - 0.5 * delta.dot(cl.stats.inv_cov).dot(delta)
        return self.classes[np.argmax(scores)].index

    def classify_image(self, image):
        '''Classifies an entire image, returning a classification map.

        Arguments:

            `image` (ndarray or :class:`spectral.Image`)

                The `MxNxB` image to classify.

        Returns (ndarray):

            An `MxN` ndarray of integers specifying the class for each pixel.
        '''
        if not (self.cache_class_scores and isinstance(image, np.ndarray)):
            return super(GaussianClassifier, self).classify_image(image)

        status = spy._status
        status.display_percentage('Processing...')
        shape = image.shape
        image = image.reshape(-1, shape[-1])
        scores = np.empty((image.shape[0], len(self.classes)), np.float64)
        delta = np.empty_like(image, dtype=np.float64)

        # For some strange reason, creating Y with np.emtpy_like will sometimes
        # result in the following error when attempting an in-place np.dot:
        #     ValueError: output array is not acceptable (must have the right
        #     type, nr dimensions, and be a C-Array)
        # It appears that this may be happening when delta is not contiguous,
        # although it isn't clear why the alternate construction of Y below
        # does work.
        Y = np.empty_like(delta)

        for (i, c) in enumerate(self.classes):
            scalar = math.log(c.class_prob) - 0.5 * c.stats.log_det_cov
            delta = np.subtract(image, c.stats.mean, out=delta)
            try:
                Y = delta.dot(-0.5 * c.stats.inv_cov, out=Y)
            except:
                # Unable to output np.dot to existing array. Allocate new
                # storage instead. This will not affect results but may be
                # slower.
                Y = delta.dot(-0.5 * c.stats.inv_cov)
            scores[:, i] = np.einsum('ij,ij->i', Y, delta)
            scores[:, i] += scalar
            status.update_percentage(100. * (i + 1) / len(self.classes))
        status.end_percentage()
        inds = np.array([c.index for c in self.classes], dtype=np.int16)
        mins = np.argmax(scores, axis=-1)
        return inds[mins].reshape(shape[:2])


class MahalanobisDistanceClassifier(GaussianClassifier):
    '''A Classifier using Mahalanobis distance for class discrimination'''
    def train(self, trainingData):
        '''Trains the classifier on the given training data.

        Arguments:

            `trainingData` (:class:`~spectral.algorithms.TrainingClassSet`):

                Data for the training classes.
        '''
        GaussianClassifier.train(self, trainingData)

        covariance = np.zeros(self.classes[0].stats.cov.shape, float)
        nsamples = np.sum(cl.stats.nsamples for cl in self.classes)
        for cl in self.classes:
            covariance += (cl.stats.nsamples / float(nsamples)) * cl.stats.cov
        self.background = GaussianStats(cov=covariance)

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
        scores = np.empty(len(self.classes))
        for (i, cl) in enumerate(self.classes):
            delta = (x - cl.stats.mean)
            scores[i] = delta.dot(self.background.inv_cov).dot(delta)
        return self.classes[np.argmin(scores)].index

    def classify_image(self, image):
        '''Classifies an entire image, returning a classification map.

        Arguments:

            `image` (ndarray or :class:`spectral.Image`)

                The `MxNxB` image to classify.

        Returns (ndarray):

            An `MxN` ndarray of integers specifying the class for each pixel.
        '''
        if not (self.cache_class_scores and isinstance(image, np.ndarray)):
            return super(MahalanobisDistanceClassifier,
                         self).classify_image(image)

        # We can cheat here and just compute RX scores for the image for each
        # class, keeping the background covariance constant and setting the
        # background mean to the mean of the particular class being evaluated.

        scores = np.empty(image.shape[:2] + (len(self.classes),), np.float64)
        status = spy._status
        status.display_percentage('Processing...')
        rx = RX()
        for (i, c) in enumerate(self.classes):
            self.background.mean = c.stats.mean
            rx.set_background(self.background)
            scores[:, :, i] = rx(image)
            status.update_percentage(100. * (i + 1) / len(self.classes))
        status.end_percentage()
        inds = np.array([c.index for c in self.classes], np.int16)
        mins = np.argmin(scores, axis=-1)
        return inds[mins]


class PerceptronClassifier(Perceptron, SupervisedClassifier):
    '''A multi-layer perceptron classifier with backpropagation learning.

    Multi-layer perceptrons often require many (i.e., thousands) of iterations
    through the training data to converge on a solution. Therefore, it is not
    recommended to attempt training a network on full-dimensional hyperspectral
    data or even on a full set of image pixels. It is likely preferable to
    first train the network on a subset of the data, then retrain the network
    (starting with network weights from initial training) on the full data
    set.

    Example usage: Train an MLP with 20 samples from each training class after
    performing dimensionality reduction:

        >>> classes = create_training_classes(data, gt)
        >>> fld = linear_discriminant(classes)
        >>> xdata = fld.transform(data)
        >>> classes = create_training_classes(xdata, gt)
        >>> nfeatures = xdata.shape[-1]
        >>> nclasses = len(classes)
        >>>
        >>> p = PerceptronClassifier([nfeatures, 20, 8, nclasses])
        >>> p.train(classes, 20, clip=0., accuracy=100., batch=1,
        >>>         momentum=0.3, rate=0.3)
        >>> c = p.classify(xdata)
    '''
    def train(self, training_data, samples_per_class=0, *args, **kwargs):
        '''Trains the Perceptron on the training data.

        Arguments:

            `training_data` (:class:`~spectral.TrainingClassSet`):

                Data for the training classes.

            `samples_per_class` (int):

                Maximum number of training observations to user from each
                class in `training_data`. If this argument is not provided,
                all training data is used.

        Keyword Arguments:

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
                suppress output, set `stdout` to None.

        Return value:

            Returns True if desired accuracy was achieved.

        Neural networks can require many iterations through a data set to
        converge. If convergence slows (as indicated by small changes in
        residual error), training can be terminated by pressing CTRL-C, which
        will preserve the network weights from the previous training iteration.
        `train` can then be called again with altered training parameters
        (e.g., increased learning rate or momentum) to increase the convergence
        rate.
        '''
        status = spy._status
        settings = spy.settings

        # Number of Perceptron inputs must equal number of features in the
        # training data.
        if len(training_data) != self.layers[-1].shape[0]:
            raise Exception('Number of nodes in output layer must match '
                            'number of training classes.')
        self.training_data = training_data

        # Map output nodes to class indices
        self.indices = [cl.index for cl in self.training_data]

        class_data = [np.array([x for x in cl]) for cl in self.training_data]
        if samples_per_class > 0:
            for i in range(len(class_data)):
                if class_data[i].shape[0] > samples_per_class:
                    class_data[i] = class_data[i][:samples_per_class]
        X = np.vstack(class_data)
        y = np.hstack([np.ones(c.shape[0], dtype=np.int16) * i for
                       (i, c) in enumerate(class_data)])
        Y = np.eye(np.max(y) + 1, dtype=np.int16)[y]

        if 'stdout' in kwargs:
            stdout = kwargs.pop('stdout')
        elif settings.show_progress is True:
            stdout = status
        else:
            stdout = None
        return Perceptron.train(self, X, Y, *args, stdout=stdout, **kwargs)

    def classify_spectrum(self, x):
        '''
        Classifies a pixel into one of the trained classes.

        Arguments:

            `x` (list or rank-1 ndarray):

                The unclassified spectrum.

        Returns:

            `classIndex` (int):

                The index for the :class:`~spectral.TrainingClass`
                to which `x` is classified.
        '''
        y = self.input(x)
        return self.indices[np.argmax(y)]

    def classify(self, X, **kwargs):
        return Classifier.classify(self, X, **kwargs)
