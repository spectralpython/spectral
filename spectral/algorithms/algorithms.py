#########################################################################
#
#   algorithms.py - This file is part of the Spectral Python (SPy)
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
Various functions and algorithms for processing spectral data.
'''
import numpy
import numpy as np

from exceptions import DeprecationWarning
from warnings import warn


class Iterator:
    '''
    Base class for iterators over pixels (spectra).
    '''
    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError('Must override __iter__ in child class.')

    def get_num_elements(self):
        raise NotImplementedError(
            'Must override get_num_elements in child class.')

    def get_num_bands(self):
        raise NotImplementedError(
            'Must override get_num_bands in child class.')


class ImageIterator(Iterator):
    '''
    An iterator over all pixels in an image.
    '''
    def __init__(self, im):
        self.image = im
        self.numElements = im.shape[0] * im.shape[1]

    def get_num_elements(self):
        return self.numElements

    def get_num_bands(self):
        return self.image.shape[2]

    def __iter__(self):
        (M, N) = self.image.shape[:2]
        count = 0
        for i in range(M):
            self.row = i
            for j in range(N):
                self.col = j
                yield self.image[i, j]


class ImageMaskIterator(Iterator):
    '''
    An iterator over all pixels in an image corresponding to a specified mask.
    '''
    def __init__(self, im, mask, index=None):
        self.image = im
        self.index = index
        # Get the proper mask for the training set
        if index:
            self.mask = numpy.equal(mask, index)
        else:
            self.mask = not_equal(mask, 0)
        self.numElements = sum(self.mask.ravel())

    def get_num_elements(self):
        return self.numElements

    def get_num_bands(self):
        return self.image.shape[2]

    def __iter__(self):
        from numpy import transpose, indices, reshape, compress, not_equal
        (nrows, ncols, nbands) = self.image.shape

        # Translate the mask into indices into the data source
        inds = transpose(indices((nrows, ncols)), (1, 2, 0))
        inds = reshape(inds, (nrows * ncols, 2))
        inds = compress(not_equal(self.mask.ravel(), 0), inds, 0).astype('h')

        for i in range(inds.shape[0]):
            sample = self.image[inds[i][0], inds[i][1]].astype(
                self.image.dtype)
            if len(sample.shape) == 3:
                sample.shape = (sample.shape[2],)
            (self.row, self.col) = inds[i][:2]
            yield sample


def iterator(image, mask=None, index=None):
    '''
    Returns an iterator over pixels in the image.

    Arguments:

        `image` (ndarray or :class:`spectral.Image`):

            An image over whose pixels will be iterated.

        `mask` (ndarray) [default None]:

            An array of integers that specify over which pixels in `image`
            iteration should be performed.

        `index` (int) [default None]:

            Specifies which value in `mask` should be used for iteration.

    Returns (:class:`spectral.Iterator`):

        An iterator over image pixels.

    If neither `mask` nor `index` are defined, iteration is performed over all
    pixels.  If `mask` (but not `index`) is defined, iteration is performed
    over all pixels for which `mask` is nonzero.  If both `mask` and `index`
    are defined, iteration is performed over all pixels `image[i,j]` for which
    `mask[i,j] == index`.
    '''

    if isinstance(image, Iterator):
        return image
    elif mask is not None:
        return ImageMaskIterator(image, mask, index)
    else:
        return ImageIterator(image)


def mean_cov(image, mask=None, index=None):
    '''
    Return the mean and covariance of the set of vectors.

    Usage::

        (mean, cov, S) = mean_cov(vectors [, mask=None [, index=None]])

    Arguments:

        `image` (ndarrray, :class:`~spectral.Image`, or :class:`spectral.Iterator`):

            If an ndarray, it should have shape `MxNxB` and the mean &
            covariance will be calculated for each band (third dimension).

        `mask` (ndarray):

            If `mask` is specified, mean & covariance will be calculated for
            all pixels indicated in the mask array.  If `index` is specified,
            all pixels in `image` for which `mask == index` will be used;
            otherwise, all nonzero elements of `mask` will be used.

        `index` (int):

            Specifies which value in `mask` to use to select pixels from
            `image`. If not specified but `mask` is, then all nonzero elements
            of `mask` will be used.

        If neither `mask` nor `index` are specified, all samples in `vectors`
        will be used.

    Returns a 3-tuple containing:

        `mean` (ndarray):

            The length-`B` mean vectors

        `cov` (ndarray):

            The `BxB` unbiased estimate (dividing by N-1) of the covariance
            of the vectors.

        `S` (int):

            Number of samples used to calculate mean & cov

    Calculate the mean and covariance of of the given vectors. The argument
    can be an Iterator, a SpyFile object, or an `MxNxB` array.
    '''
    import spectral
    import numpy as np
    from numpy import zeros, transpose, dot, newaxis

    status = spectral._status

    if isinstance(image, np.ndarray):
        X = image.astype(np.float64)
        if X.ndim == 3:
            X = image.reshape(-1, image.shape[-1]).T
        if mask is not None:
            mask = mask.ravel()
            if index is not None:
                ii = np.argwhere(mask == index)
            else:
                ii = np.argwhere(mask != 0)
            X = np.take(X, ii.squeeze(), axis=1)
        m = np.average(X, axis=1)
        C = np.cov(X)
        return (m, C, X.shape[1])

    if not isinstance(image, Iterator):
        it = iterator(image, mask, index)
    else:
        it = image

    nSamples = it.get_num_elements()
    B = it.get_num_bands()

    sumX = zeros((B,), 'd')
    sumX2 = zeros((B, B), 'd')
    count = 0

    statusInterval = max(1, nSamples / 100)
    status.display_percentage('Covariance.....')
    for x in it:
        if not count % statusInterval:
            status.update_percentage(float(count) / nSamples * 100.)
        count += 1
        sumX += x
        x = x.astype(np.float64)[:, newaxis]
        sumX2 += x.dot(x.T)
    mean = (sumX / count)
    sumX = sumX[:, newaxis]
    cov = (sumX2 - sumX.dot(sumX.T) / count) / (count - 1)
    status.end_percentage()
    return (mean, cov, count)


def covariance(*args):
    '''
    Returns the covariance of the set of vectors.

    Usage::

        C = covariance(vectors [, mask=None [, index=None]])

    Arguments:

        `vectors` (ndarrray, :class:`~spectral.Image`, or :class:`spectral.Iterator`):

            If an ndarray, it should have shape `MxNxB` and the mean &
            covariance will be calculated for each band (third dimension).

        `mask` (ndarray):

            If `mask` is specified, mean & covariance will be calculated for
            all pixels indicated in the mask array.  If `index` is specified,
            all pixels in `image` for which `mask == index` will be used;
            otherwise, all nonzero elements of `mask` will be used.

        `index` (int):

            Specifies which value in `mask` to use to select pixels from
            `image`. If not specified but `mask` is, then all nonzero elements
            of `mask` will be used.

        If neither `mask` nor `index` are specified, all samples in `vectors`
        will be used.

    Returns:

        `C` (ndarray):

            The `BxB` unbiased estimate (dividing by N-1) of the covariance
            of the vectors.


    To also return the mean vector and number of samples, call
    :func:`~spectral.algorithms.algorithms.mean_cov` instead.
    '''
    return mean_cov(*args)[1]


class PrincipalComponents:
    '''
    An object for storing a data set's principal components.  The
    object has the following members:

        `eigenvalues`:

            A length B array of eigenvalues

        `eigenvectors`:

            A `BxB` array of normalized eigenvectors

        `stats` (:class:`GaussianStats`):

            A statistics object containing `mean`, `cov`, and `nsamples`.

        `transform`:

            A callable function to transform data to the space of the
            principal components.

        `reduce`:

            A method to return a reduced set of principal components based
            on either a fixed number of components or a fraction of total
            variance.
    '''
    def __init__(self, vals, vecs, stats):
        from transforms import LinearTransform
        self.eigenvalues = vals
        self.eigenvectors = vecs
        self.stats = stats
        self.transform = LinearTransform(self.eigenvectors, pre=-self.mean)

    @property
    def mean(self):
        return self.stats.mean

    @property
    def cov(self):
        return self.stats.cov

    def reduce(self, N=0, **kwargs):
        '''Reduces the number of principal components.

        Keyword Arguments (one of the following must be specified):

            `num` (integer):

                Number of eigenvalues/eigenvectors to retain.  The top `num`
                eigenvalues will be retained.

            `eigs` (list):

                A list of indices of eigenvalues/eigenvectors to be retained.

            `fraction` (float):

                The fraction of total image variance to retain.  Eigenvalues
                will be retained (starting from greatest to smallest) until
                `fraction` of total image variance is retained.
        '''
        import spectral

        status = spectral._status

        num = kwargs.get('num', None)
        eigs = kwargs.get('eigs', None)
        fraction = kwargs.get('fraction', None)
        if num is not None:
            return PrincipalComponents(self.eigenvalues[:num],
                                       self.eigenvectors[:num],
                                       self.stats)
        elif eigs is not None:
            vals = self.eigenvalues[eigs]
            vecs = self.eigenvectors[eigs]
            return PrincipalComponents(vals, vecs, self.stats)
        elif fraction is not None:
            if not 0 < fraction <= 1:
                raise Exception('fraction must be in range (0,1].')
            N = len(self.eigenvalues)
            cumsum = numpy.cumsum(self.eigenvalues)
            sum = cumsum[-1]
            # Count how many values to retain.
            for i in range(N):
                if (cumsum[i] / sum) >= fraction:
                    break
            if i == (N - 1):
                # No reduction
                status.write('No reduction in eigenvectors achieved.')
                return self

            vals = self.eigenvalues[:i + 1]
            vecs = self.eigenvectors[:i + 1, :]
            return PrincipalComponents(vals, vecs, self.stats)
        else:
            raise Exception('Must specify one of the following keywords:'
                            '`num`, `eigs`, `fraction`.')


def principal_components(image):
    '''
    Calculate Principal Component eigenvalues & eigenvectors for an image.

    Usage::

        pc = principal_components(image)

    Arguments:

        `image` (ndarray or :class:`spectral.Image`):

            An `MxNxB` image

    Returns a :class:`~spectral.algorithms.algorithms.PrincipalComponents`
    object with the following members:

        `eigenvalues`:

            A length B array of eigenvalues

        `eigenvectors`:

            A `BxB` array of normalized eigenvectors

        `stats` (:class:`GaussianStats`):

            A statistics object containing `mean`, `cov`, and `nsamples`.

        `transform`:

            A callable function to transform data to the space of the
            principal components.

        `reduce`:

            A method to reduce the number of eigenvalues.
    '''
    from numpy import sqrt, sum

    (M, N, B) = image.shape

    stats = calc_stats(image)

    (L, V) = numpy.linalg.eig(stats.cov)

    #  Normalize eigenvectors
    V = V / sqrt(sum(V * V, 0))

    # numpy stores eigenvectors in columns
    V = V.transpose()

    return PrincipalComponents(L, V, stats)


class FisherLinearDiscriminant:
    '''
    An object for storing a data set's principal components.  The
    object has the following members:

        `eigenvalues`:

            A length B array of eigenvalues

        `eigenvectors`:

            A `BxB` array of normalized eigenvectors

        `mean`:

            The length `B` mean vector of the image pixels (from all classes)

        `cov_b`:

            The `BxB` matrix of covariance *between* classes

        `cov_w`:

            The `BxB` matrix of average covariance *within* each class

        `transform`:

            A callable function to transform data to the space of the
            linear discriminant.
    '''
    def __init__(self, vals, vecs, mean, cov_b, cov_w):
        from transforms import LinearTransform
        self.eigenvalues = vals
        self.eigenvectors = vecs
        self.mean = mean
        self.cov_b = cov_b
        self.cov_w = cov_w
        self.transform = LinearTransform(self.eigenvectors, pre=-self.mean)


def linear_discriminant(classes):
    '''
    Solve Fisher's linear discriminant for eigenvalues and eigenvectors.

    Usage: (L, V, Cb, Cw) = linear_discriminant(classes)

    Arguments:

        `classes` (:class:`~spectral.algorithms.TrainingClassSet`):

            The set of `C` classes to discriminate.

    Returns a `FisherLinearDiscriminant` object containing the within/between-
    class covariances, mean vector, and a callable transform to convert data to
    the transform's space.

    This function determines the solution to the generalized eigenvalue problem

            Cb * x = lambda * Cw * x

    Since cov_w is normally invertable, the reduces to

            (inv(Cw) * Cb) * x = lambda * x

    References:

        Richards, J.A. & Jia, X. Remote Sensing Digital Image Analysis: An
        Introduction. (Springer: Berlin, 1999).
    '''

    from numpy import zeros, dot, transpose, diagonal
    from numpy.linalg import inv, eig
    from numpy.oldnumeric import NewAxis
    import math

    C = len(classes)            # Number of training sets
    rank = len(classes) - 1

    # Calculate total # of training pixels and total mean
    N = 0
    B = None            # Don't know number of bands yet
    mean = None
    for s in classes:
        if mean is None:
            B = s.nbands
            mean = zeros(B, float)
        N += s.size()
        if not hasattr(s, 'stats'):
            s.calc_stats()
        mean += s.size() * s.stats.mean
    mean /= float(N)

    cov_b = zeros((B, B), float)            # cov between classes
    cov_w = zeros((B, B), float)            # cov within classes

    for s in classes:
        cov_w += (s.size() - 1) * s.stats.cov
        m = (s.stats.mean - mean)[:, NewAxis]
        cov_b += s.size() * dot(m, transpose(m))
    cov_w /= float(N)
    cov_b /= float(N)

    cwInv = inv(cov_w)
    (vals, vecs) = eig(dot(cwInv, cov_b))

    vals = vals[:rank]
    vecs = transpose(vecs)[:rank, :]

    # Diagonalize cov_within in the new space
    v = dot(vecs, dot(cov_w, transpose(vecs)))
#    d = diagonal(v)
    d = numpy.sqrt(diagonal(v) * diagonal(v).conj())
    for i in range(vecs.shape[0]):
        vecs[i, :] /= math.sqrt(d[i].real)

    return FisherLinearDiscriminant(vals.real, vecs.real, mean, cov_b, cov_w)

# Alias for Linear Discriminant Analysis (LDA)
lda = linear_discriminant


def log_det(x):
    return sum(numpy.log([eigv for eigv in numpy.linalg.eigvals(x)
                          if eigv > 0]))


class GaussianStats:

    def __init__(self, **kwargs):
        self.nsamples = kwargs.get('nsamples', 0)
        self.mean = kwargs.get('mean', None)
        self.cov = kwargs.get('cov', None)

    def transform(self, xform):
        '''Returns a version of the stats transformed by a linear transform.'''
        from spectral.algorithms.transforms import LinearTransform
        if not isinstance(xform, LinearTransform):
            raise TypeError('Expected a LinearTransform object.')
        m = xform(self.mean)
        C = xform._A.dot(self.cov).dot(xform._A.T)
        return GaussianStats(mean=m, cov=C, nsamples=self.nsamples)

    def get_whitening_transform(self):
        '''Returns transform that centers and whitens data for these stats.'''
        from spectral.algorithms.transforms import LinearTransform
        from spectral.algorithms.spymath import matrix_sqrt
        C_1 = np.linalg.inv(self.cov)
        return LinearTransform(matrix_sqrt(C_1, True), pre=-self.mean)


def calc_stats(image, mask=None, index=None):
    '''Computes Gaussian stats for image data..

    Arguments:

        `image` (ndarrray, :class:`~spectral.Image`, or :class:`spectral.Iterator`):

            If an ndarray, it should have shape `MxNxB` and the mean &
            covariance will be calculated for each band (third dimension).

        `mask` (ndarray):

            If `mask` is specified, mean & covariance will be calculated for
            all pixels indicated in the mask array.  If `index` is specified,
            all pixels in `image` for which `mask == index` will be used;
            otherwise, all nonzero elements of `mask` will be used.

        `index` (int):

            Specifies which value in `mask` to use to select pixels from
            `image`. If not specified but `mask` is, then all nonzero elements
            of `mask` will be used.

        If neither `mask` nor `index` are specified, all samples in `vectors`
        will be used.

    Returns:

        `GaussianStats` object:

            This object will have members `mean`, `cov`, and `nsamples`.
    '''
    (mean, cov, N) = mean_cov(image, mask, index)
    return GaussianStats(mean=mean, cov=cov, nsamples=N)


class TrainingClass:
    def __init__(self, image, mask, index=0, class_prob=1.0):
        '''Creates a new training class defined by applying `mask` to `image`.

        Arguments:

            `image` (:class:`spectral.Image` or :class:`numpy.ndarray`):

                The `MxNxB` image over which the training class is defined.

            `mask` (:class:`numpy.ndarray`):

                An `MxN` array of integers that specifies which pixels in
                `image` are associated with the class.

            `index` (int) [default 0]:

                if `index` == 0, all nonzero elements of `mask` are associated
                with the class.  If `index` is nonzero, all elements of `mask`
                equal to `index` are associated with the class.

            `classProb` (float) [default 1.0]:

                Defines the prior probability associated with the class, which
                is used in maximum likelihood classification.  If `classProb`
                is 1.0, prior probabilities are ignored by classifiers, giving
                all class equal weighting.
        '''
        self.image = image
        self.nbands = image.shape[2]
        self.mask = mask
        self.index = index
        self.class_prob = class_prob

        self._stats_valid = 0
        self._size = 0

    def __iter__(self):
        '''Returns an iterator over all samples for the class.'''
        it = ImageMaskIterator(self.image, self.mask, self.index)
        for i in it:
            yield i

    def stats_valid(self, tf):
        '''
        Sets statistics for the TrainingClass to be valid or invalid.

        Arguments:

            `tf` (bool):

                A value evaluating to True indicates that statistics should be
                recalculated prior to being used.
        '''
        self._stats_valid = tf

    def size(self):
        '''Returns the number of pixels/samples in the training set.'''
        from numpy import sum, equal

        # If the stats are invalid, the number of pixels in the
        # training set may have changed.
        if self._stats_valid:
            return self._size

        if self.index:
            return sum(equal(self.mask, self.index).ravel())
        else:
            return sum(not_equal(self.mask, 0).ravel())

    def calc_stats(self):
        '''
        Calculates statistics for the class.

        This function causes the :attr:`stats` attribute of the class to be
        updated, where `stats` will have the following attributes:

        =============  ======================   ===================================
        Attribute      Type                          Description
        =============  ======================   ===================================
        `mean`         :class:`numpy.ndarray`   length-`B` mean vector
        `cov`          :class:`numpy.ndarray`   `BxB` covariance matrix
        `inv_cov`      :class:`numpy.ndarray`   Inverse of `cov`
        `log_det_cov`  float                    Natural log of determinant of `cov`
        =============  ======================   ===================================
        '''
        import math
        from numpy.linalg import inv, det

        self.stats = calc_stats(self.image, self.mask, self.index)
        self.stats.inv_cov = inv(self.stats.cov)
        self.stats.log_det_cov = log_det(self.stats.cov)
        self._size = self.stats.nsamples
        self._stats_valid = 1

    def transform(self, transform):
        '''
        Perform a linear transformation on the statistics of the training set.

        Arguments:

            `transform` (:class:numpy.ndarray or LinearTransform):

                The linear transform array.  If the class has `B` bands, then
                `transform` must have shape `(C,B)`.

        After `transform` is applied, the class statistics will have `C` bands.
        '''

        from transforms import LinearTransform
        from numpy.linalg import det, inv
        import math
        from spectral.io.spyfile import TransformedImage

        if isinstance(transform, np.ndarray):
            transform = LinearTransform(transform)
        self.stats.mean = transform(self.stats.mean)
        self.stats.cov = np.dot(
            transform._A, self.stats.cov).dot(transform._A.T)
        self.stats.inv_cov = inv(self.stats.cov)
        self.nbands = transform.dim_out

        self.stats.log_det_cov = log_det(self.stats.cov)

    def dump(self, fp):
        '''
        Dumps the TrainingClass object to a file stream.  Note that the
        image reference is replaced by the images file name.  It the
        responsibility of the loader to verify that the file name
        is replaced with an actual image object.
        '''
        import pickle

        pickle.dump(self.image.file_name, fp)
        pickle.dump(self.index, fp)
        pickle.dump(self._size, fp)
        pickle.dump(self.classProb, fp)
        DumpArray(self.mask, fp)
        DumpArray(self.stats.mean, fp)
        DumpArray(self.stats.cov, fp)
        DumpArray(self.stats.inv_cov, fp)
        pickle.dump(self.stats.log_det_cov, fp)

    def load(self, fp):
        '''
        Loads the TrainingClass object from a file stream.  The image
        member was probably replaced by the name of the image's source
        file before serialization.  The member should be replaced by
        the caller with an actual image object.
        '''
        import pickle

        self.stats = GaussianStats()

        self.image = pickle.load(fp)
        self.index = pickle.load(fp)
        self._size = pickle.load(fp)
        self.class_prob = pickle.load(fp)
        self.mask = LoadArray(fp)
        self.stats.mean = LoadArray(fp)
        self.stats.cov = LoadArray(fp)
        self.stats.inv_cov = LoadArray(fp)
        self.stats.log_det_cov = pickle.load(fp)
        self.stats.num_samples = self._size

    # Deprecated methods
    def calcStatistics(self):
        warn('TrainingClass.calcStatistics has been deprecated. '
             + 'Use TrainingClass.calc_stats.', DeprecationWarning)
        return self.calc_stats()


class SampleIterator:
    '''Iterator over all classes and samples in a TrainingClassSet object.'''
    def __init__(self, trainingData):
        self.classes = trainingData

    def __iter__(self):
        for cl in self.classes:
            for sample in cl:
                yield sample


class TrainingClassSet:
    '''A class to manage a set of :class:`~spectral.TrainingClass` objects.'''
    def __init__(self):
        self.classes = {}
        self.nbands = None

    def __getitem__(self, i):
        '''Returns the training class having ID i.'''
        return self.classes[i]

    def __len__(self):
        '''Returns number of training classes in the set.'''
        return len(self.classes)

    def add_class(self, cl):
        '''Adds a new class to the training set.

        Arguments:

            `cl` (:class:`spectral.TrainingClass`):

                `cl.index` must not duplicate a class already in the set.
        '''
        if cl.index in self.classes:
            raise Exception('Attempting to add class with duplicate index.')
        self.classes[cl.index] = cl
        if not self.nbands:
            self.nbands = cl.nbands

    def transform(self, X):
        '''Applies linear transform, M, to all training classes.

        Arguments:

            `X` (:class:numpy.ndarray):

                The linear transform array.  If the classes have `B` bands, then
                `X` must have shape `(C,B)`.

        After the transform is applied, all classes will have `C` bands.
        '''
        for cl in self.classes.values():
            cl.transform(X)
        self.nbands = self.classes.values()[0].nbands

    def __iter__(self):
        '''An iterator over all training classes in the set.'''
        for cl in self.classes.values():
            yield cl

    def all_samples(self):
        '''An iterator over all samples in all classes.'''
        return SampleIterator(self)

    #-------------------
    # Deprecated methods
    #-------------------
    def addClass(self, cl):
        '''DEPRECATED METHOD'''
        warn('TrainingClassSet.addClass has been deprecated. '
             + 'Use TrainingClassSet.add_class.', DeprecationWarning)
        return self.add_class(cl)

    def allSamples(self):
        '''DEPRECATED METHOD'''
        warn('TrainingClassSet.calcStatistics has been deprecated. '
             + 'Use TrainingClassSet.all_samples.', DeprecationWarning)
        return self.all_samples()


def create_training_classes(image, class_mask, calc_stats=0, indices=None):
    '''
    Creates a :class:spectral.algorithms.TrainingClassSet: from an indexed array.

    USAGE:  sets = createTrainingClasses(classMask)

    Arguments:

        `image` (:class:`spectral.Image` or :class:`numpy.ndarray`):

            The image data for which the training classes will be defined.
            `image` has shape `MxNxB`.

        `class_mask` (:class:`numpy.ndarray`):

            A rank-2 array whose elements are indices of various spectral
            classes.  if `class_mask[i,j]` == `k`, then `image[i,j]` is
            assumed to belong to class `k`.

        `calc_stats`:

            An optional parameter which, if True, causes statistics to be
            calculated for all training classes.

    Returns:

        A :class:`spectral.algorithms.TrainingClassSet` object.

    The dimensions of classMask should be the same as the first two dimensions
    of the corresponding image. Values of zero in classMask are considered
    unlabeled and are not added to a training set.
    '''

    class_indices = set(class_mask.ravel())
    classes = TrainingClassSet()
    for i in class_indices:
        if i == 0:
            # Index 0 denotes unlabled pixel
            continue
        elif indices and not i in indices:
            continue
        cl = TrainingClass(image, class_mask, i)
        if calc_stats:
            cl.calc_stats()
        classes.add_class(cl)
    return classes


def ndvi(data, red, nir):
    '''Calculates Normalized Difference Vegetation Index (NDVI).

    Arguments:

        `data` (ndarray or :class:`spectral.Image`):

            The array or SpyFile for which to calculate the index.

        `red` (int or int range):

            Index of the red band or an index range for multiple bands.

        `nir` (int or int range):

            An integer index of the near infrared band or an index range for
            multiple bands.

    Returns an ndarray:

        An array containing NDVI values in the range [-1.0, 1.0] for each
        corresponding element of data.
    '''

    r = data[:, :, red].astype(float)
    if len(r.shape) == 3 and r.shape[2] > 1:
        r = sum(r, 2) / r.shape[2]
    n = data[:, :, nir].astype(float)
    if len(n.shape) == 3 and n.shape[2] > 1:
        n = sum(n, 2) / n.shape[2]

    return (n - r) / (n + r)


def bdist(class1, class2):
    '''
    Calulates the Bhattacharyya distance between two classes.

    USAGE:  bd = bdist(class1, class2)

    Arguments:

        `class1`, `class2` (:class:`~spectral.algorithms.algorithms.TrainingClass`)

    Returns:

        A float value for the Bhattacharyya Distance between the classes.  This
        function is aliased to :func:`~spectral.algorithms.algorithms.bDistance`.

    References:

        Richards, J.A. & Jia, X. Remote Sensing Digital Image Analysis: An
        Introduction. (Springer: Berlin, 1999).
    '''
    terms = bdist_terms(class1, class2)
    return terms[0] + terms[1]

bDistance = bdist


def bdist_terms(a, b):
    '''
    Calulate the linear and quadratic terms of the Bhattacharyya distance
    between two classes.

    USAGE:  (linTerm, quadTerm) = bDistanceTerms(a, b)

    ARGUMENTS:
        (a, b)              The classes for which to determine the
                            B-distance.
    RETURN VALUE:
                            A 2-tuple of the linear and quadratic terms
    '''
    from math import exp
    from numpy import dot, transpose
    from numpy.linalg import inv

    m = a.stats.mean - b.stats.mean
    avgCov = (a.stats.cov + b.stats.cov) / 2

    lin_term = (1 / 8.) * dot(transpose(m), dot(inv(avgCov), m))

    quad_term = 0.5 * (log_det(avgCov)
                       - 0.5 * a.stats.log_det_cov
                       - 0.5 * b.stats.log_det_cov)

    return (lin_term, float(quad_term))


def transform_image(matrix, image):
    '''
    Performs linear transformation on all pixels in an image.

    Arguments:

        matrix (:class:`numpy.ndarray`):

            A `CxB` linear transform to apply.

        image  (:class:`numpy.ndarray` or :class:`spectral.Image`):

            Image data to transform

    Returns:

        If `image` is an `MxNxB` :class:`numpy.ndarray`, the return will be a
        transformed :class:`numpy.ndarray` with shape `MxNxC`.  If `image` is
        :class:`spectral.Image`, the returned object will be a
        :class:`spectral.TransformedImage` object and no transformation of data
        will occur until elements of the object are accessed.
    '''
    from spectral.io.spyfile import TransformedImage
    from numpy.oldnumeric import ArrayType
    from spectral.io.spyfile import SpyFile

    if isinstance(image, SpyFile):
        return TransformedImage(matrix, image)
    elif isinstance(image, ArrayType):
        (M, N, B) = image.shape
        ximage = numpy.zeros((M, N, matrix.shape[0]), float)

        for i in range(M):
            for j in range(N):
                ximage[i, j] = numpy.dot(matrix, image[i, j].astype(float))
        return ximage
    else:
        raise 'Unrecognized image type passed to transform_image.'


def orthogonalize(vecs, start=0):
    '''
    Performs Gram-Schmidt Orthogonalization on a set of vectors.

    Arguments:

        `vecs` (:class:`numpy.ndarray`):

            The set of vectors for which an orthonormal basis will be created.
            If there are `C` vectors of length `B`, `vecs` should be `CxB`.

        `start` (int) [default 0]:

            If `start` > 0, then `vecs[start]` will be assumed to already be
            orthonormal.

    Returns:

        A new `CxB` containing an orthonormal basis for the given vectors.
    '''

    from numpy import transpose, dot, identity
    from numpy.linalg import inv
    from math import sqrt

    (M, N) = vecs.shape
    basis = numpy.array(transpose(vecs))
    eye = identity(N).astype(float)
    if start == 0:
        basis[:, 0] /= sqrt(dot(basis[:, 0], basis[:, 0]))
        start = 1

    for i in range(start, M):
        v = basis[:, i] / sqrt(dot(basis[:, i], basis[:, i]))
        U = basis[:, :i]
        P = eye - dot(U, dot(inv(dot(transpose(U), U)), transpose(U)))
        basis[:, i] = dot(P, v)
        basis[:, i] /= sqrt(dot(basis[:, i], basis[:, i]))

    return transpose(basis)


def unmix(data, members):
    '''
    Perform linear unmixing on image data.

    USAGE: mix = unmix(data, members)

    ARGUMENTS:
        data                The MxNxB image data to be unmixed
        members             An CxB array of C endmembers
    RETURN VALUE:
        mix                 An MxNxC array of endmember fractions.

    unmix performs linear unmixing on the image data.  After calling the
    function, mix[:,:,i] will then represent the fractional abundances
    for the i'th endmember. If the result of unmix is returned into 'mix',
    then an array of indices of greatest fractional endmembers is obtained
    by argmax(mix).

    Note that depending on endmembers given, fractional abundances for
    endmembers may be negative.
    '''

    from numpy import transpose, dot, zeros
    from numpy.linalg import inv

    assert members.shape[1] == data.shape[2], \
        'Matrix dimensions are not aligned.'

    members = members.astype(float)
    # Calculate the pseudo inverse
    pi = dot(members, transpose(members))
    pi = dot(inv(pi), members)

    (M, N, B) = data.shape
    unmixed = zeros((M, N, members.shape[0]), float)
    for i in range(M):
        for j in range(N):
            unmixed[i, j] = dot(pi, data[i, j].astype(float))
    return unmixed


def spectral_angles(data, members):
    '''Calculates spectral angles with respect to given set of spectra.

    Arguments:

        `data` (:class:`numpy.ndarray` or :class:`spectral.Image`):

            An `MxNxB` image for which spectral angles will be calculated.

        `members` (:class:`numpy.ndarray`):

            `CxB` array of spectral endmembers.

    Returns:

        `MxNxC` array of spectral angles.


    Calculates the spectral angles between each vector in data and each of the
    endmembers.  The output of this function (angles) can be used to classify
    the data by minimum spectral angle by calling argmin(angles).
    '''
    from numpy import array, dot, zeros, arccos, sqrt

    assert members.shape[1] == data.shape[2], \
        'Matrix dimensions are not aligned.'

    (M, N, B) = data.shape
    m = array(members, float)
    C = m.shape[0]

    # Normalize endmembers
    for i in range(C):
        m[i] /= sqrt(dot(m[i], m[i]))

    angles = zeros((M, N, C), float)

    for i in range(M):
        for j in range(N):
            v = data[i, j].astype(float)
            v = v / sqrt(dot(v, v))
            for k in range(C):
                angles[i, j, k] = dot(v, m[k])

    return arccos(angles)

#---------------------
# Deprecated functions
#---------------------


def principalComponents(image):
    warn('principalComponents has been deprecated.  Use principal_components.',
         DeprecationWarning)
    return principal_components(image)


def linearDiscriminant(classes):
    warn('linearDiscriminant has been deprecated.  Use linear_discriminant.',
         DeprecationWarning)
    return linear_discriminant(classes)


def reduceEigenvectors(L, V, fraction=0.99):
    warn('reduceEigenvectors has been deprecated.  Use reduce_eigenvectors.',
         DeprecationWarning)
    return reduce_eigenvectors(L, V, fraction)


def logDeterminant(x):
    warn('logDeterminant has been deprecated.  Use log_det.',
         DeprecationWarning)
    return log_det(x)


def createTrainingClasses(image, class_mask, calc_stats=0, indices=None):
    warn('createTrainingClasses has been deprecated. '
         + 'Use create_training_classes.', DeprecationWarning)
    return create_training_classes(image, class_mask, calc_stat, indices)


def bhattacharyyaDistance(class1, class2):
    warn('bhattacharyyaDistance has been deprecated.  Use bdist.',
         DeprecationWarning)
    return bdist(class1, class2)


def bDistanceTerms(a, b):
    warn('bDistanceTerms has been deprecated.  Use bdist_terms.',
         DeprecationWarning)
    return bdist_terms(image)


def transformImage(matrix, image):
    warn('transform_image has been deprecated.  Use transform_image.',
         DeprecationWarning)
    return transform_image(matrix, image)


def spectralAngles(data, members):
    warn('spectralAngles has been deprecated.  Use spectral_angles.',
         DeprecationWarning)
    return spectral_angles(data, members)
