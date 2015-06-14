#########################################################################
#
#   clustering.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2001-2011 Thomas Boggs
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
Unsupervised clustering algorithms.
'''

from __future__ import division, print_function, unicode_literals

import numpy
from .classifiers import Classifier

from warnings import warn


def L1(v1, v2):
    'Returns L1 distance between 2 rank-1 arrays.'
    return numpy.sum(abs((v1 - v2)))


def L2(v1, v2):
    'Returns Euclidean distance between 2 rank-1 arrays.'
    delta = v1 - v2
    return numpy.sqrt(numpy.dot(delta, delta))


class KmeansClusterer(Classifier):
    '''An unsupervised classifier using an iterative clustering algorithm'''
    def __init__(self, nclusters=10, maxIter=20, endCondition=None,
                 distanceMeasure=L1):
        '''
        ARGUMENTS:
            nclusters       Number of clusters to create. Default is 8
            maxIter         Max number of iterations. Default is 20
            endCondition    Optional comparison function. This should be a
                            function which takes 2 MxN NumPy arrays as its
                            arguments and returns non-zero when clustering
                            is to be terminated. The two arguments are the
                            cluster maps for the previous and current cluster
                            cycle, respectively.
            distanceMeasure The distance measure to use for comparison. The
                            default is the L1 distance. For  Euclidean
                            distance, specify L2 (no quotes).
        '''
        self.nclusters = nclusters
        self.maxIterations = maxIter
        self.endCondition = endCondition
        self.distanceMeasure = distanceMeasure

    def classify_image(self, image, startClusters=None, iterations=None):
        '''
        Performs iterative self-organizing clustering of image data.

        USAGE: (clMap, centers) = cl.classify_image(image
                                                   [, startClusters = None]
                                                   [, iterations = None])

        ARGUMENTS:
            image           A SpyFile or an MxNxB NumPy array
            startClusters   Initial cluster centers. This must be an
                            nclusters x B array.
            iterations      If this argument is passed and is a list object,
                            each intermediate cluster map is appended to
                            the list.
        RETURN VALUES:
            clMap           An MxN array whos values are the indices of the
                            cluster for the corresponding element of image.
            centers         An nclusters x B array of cluster centers.
        '''
        return isoCluster(
            image, self.nclusters, self.maxIterations, startClusters,
            self.endCondition, self.distanceMeasure, iterations)


def kmeans(image, nclusters=10, max_iterations=20, **kwargs):
    '''
    Performs iterative clustering using the k-means algorithm.

    Arguments:

        `image` (:class:`numpy.ndarray` or :class:`spectral.Image`):

            The `MxNxB` image on which to perform clustering.

        `nclusters` (int) [default 10]:

            Number of clusters to create.  The number produced may be less than
            `nclusters`.

        `max_iterations` (int) [default 20]:

            Max number of iterations to perform.

    Keyword Arguments:

        `start_clusters` (:class:`numpy.ndarray`) [default None]:

            `nclusters x B` array of initial cluster centers.  If not provided,
            initial cluster centers will be spaced evenly along the diagonal of
            the N-dimensional bounding box of the image data.

        `compare` (callable object) [default None]:

            Optional comparison function. `compare` must be a callable object
            that takes 2 `MxN` :class:`numpy.ndarray` objects as its arguments
            and returns non-zero when clustering is to be terminated. The two
            arguments are the cluster maps for the previous and current cluster
            cycle, respectively.

        `distance` (callable object) [default :func:`~spectral.clustering.L2`]:

            The distance measure to use for comparison. The default is to use
            **L2** (Euclidean) distance. For Manhattan distance, specify
            :func:`~spectral.clustering.L1`.

        `frames` (list) [default None]:

            If this argument is given and is a list object, each intermediate
            cluster map is appended to the list.

    Returns a 2-tuple containing:

        `class_map` (:class:`numpy.ndarray`):

            An `MxN` array whos values are the indices of the cluster for the
            corresponding element of `image`.

        `centers` (:class:`numpy.ndarray`):

            An `nclusters x B` array of cluster centers.

    Iterations are performed until clusters converge (no pixels reassigned
    between iterations), `maxIterations` is reached, or `compare` returns
    nonzero. If :exc:`KeyboardInterrupt` is generated (i.e., CTRL-C pressed)
    while the algorithm is executing, clusters are returned from the previously
    completed iteration.
    '''
    import spectral
    import numpy

    if isinstance(image, numpy.ndarray):
        return kmeans_ndarray(*(image, nclusters, max_iterations), **kwargs)

    status = spectral._status

    # defaults for kwargs
    start_clusters = None
    compare = None
    distance = L2
    iterations = None

    for (key, val) in list(kwargs.items()):
        if key == 'start_clusters':
            start_clusters = val
        elif key == 'compare':
            compare = val
        elif key == 'distance':
            if val in (L1, 'L1'):
                distance = L1
            elif val in (L2, 'L2'):
                distance = L2
            else:
                raise ValueError('Unrecognized keyword argument.')
        elif key == 'frames':
            if not hasattr(val, 'append'):
                raise TypeError('"frames" keyword argument must have "append"'
                                'attribute.')
            iterations = frames
        else:
            raise NameError('Unsupported keyword argument.')

    (nrows, ncols, nbands) = image.shape
    clusters = numpy.zeros((nrows, ncols), int)
    old_clusters = numpy.copy(clusters)
    if start_clusters is not None:
        assert (start_clusters.shape[0] == nclusters), 'There must be \
        nclusters clusters in the startCenters array.'
        centers = numpy.array(start_clusters)
    else:
        print('Initializing clusters along diagonal of N-dimensional bounding box.')
        centers = numpy.empty((nclusters, nbands), float)
        boxMin = image[0, 0]
        boxMax = image[0, 0]
        for i in range(nrows):
            for j in range(ncols):
                x = image[i, j]
                boxMin = numpy.where(boxMin < x, boxMin, x)
                boxMax = numpy.where(boxMax > x, boxMax, x)
        boxMin = boxMin.astype(float)
        boxMax = boxMax.astype(float)
        delta = (boxMax - boxMin) / (nclusters - 1)
        for i in range(nclusters):
            centers[i] = boxMin.astype(float) + i * delta

    itnum = 1
    while (itnum <= max_iterations):
        try:
            status.display_percentage('Iteration %d...' % itnum)

            # Assign all pixels
            for i in range(nrows):
                status.update_percentage(float(i) / nrows * 100.)
                for j in range(ncols):
                    minDist = 1.e30
                    for k in range(nclusters):
                        dist = distance(image[i, j], centers[k])
                        if (dist < minDist):
                            clusters[i, j] = k
                            minDist = dist

            # Update cluster centers
            sums = numpy.zeros((nclusters, nbands), 'd')
            counts = ([0] * nclusters)
            for i in range(nrows):
                for j in range(ncols):
                    counts[clusters[i, j]] += 1
                    sums[clusters[i, j]] += image[i, j]

            old_centers = centers[:]
            for i in range(nclusters):
                if (counts[i] > 0):
                    centers[i] = sums[i] / counts[i]
            centers = numpy.array(centers)

            if iterations is not None:
                iterations.append(clusters)

            if compare and compare(old_clusters, clusters):
                status.end_percentage('done.')
                break
            else:
                nChanged = numpy.sum(clusters != old_clusters)
                if nChanged == 0:
                    status.end_percentage('0 pixels reassigned.')
                    break
                else:
                    status.end_percentage('%d pixels reassigned.' \
                                          % (nChanged))

            old_clusters = clusters
            old_centers = centers
            clusters = numpy.zeros((nrows, ncols), int)
            itnum += 1

        except KeyboardInterrupt:
            print("KeyboardInterrupt: Returning clusters from previous iteration")
            return (old_clusters, old_centers)

    print('kmeans terminated with', len(set(old_clusters.ravel())), \
        'clusters after', itnum - 1, 'iterations.', file=status)
    return (old_clusters, centers)


def kmeans_ndarray(image, nclusters=10, max_iterations=20, **kwargs):
    '''
    Performs iterative clustering using the k-means algorithm.

    Arguments:

        `image` (:class:`numpy.ndarray` or :class:`spectral.Image`):

            The `MxNxB` image on which to perform clustering.

        `nclusters` (int) [default 10]:

            Number of clusters to create.  The number produced may be less than
            `nclusters`.

        `max_iterations` (int) [default 20]:

            Max number of iterations to perform.

    Keyword Arguments:

        `start_clusters` (:class:`numpy.ndarray`) [default None]:

            `nclusters x B` array of initial cluster centers.  If not provided,
            initial cluster centers will be spaced evenly along the diagonal of
            the N-dimensional bounding box of the image data.

        `compare` (callable object) [default None]:

            Optional comparison function. `compare` must be a callable object
            that takes 2 `MxN` :class:`numpy.ndarray` objects as its arguments
            and returns non-zero when clustering is to be terminated. The two
            arguments are the cluster maps for the previous and current cluster
            cycle, respectively.

        `distance` (callable object) [default :func:`~spectral.clustering.L2`]:

            The distance measure to use for comparison. The default is to use
            **L2** (Euclidean) distance. For Manhattan distance, specify
            :func:`~spectral.clustering.L1`.

        `frames` (list) [default None]:

            If this argument is given and is a list object, each intermediate
            cluster map is appended to the list.

    Returns a 2-tuple containing:

        `class_map` (:class:`numpy.ndarray`):

            An `MxN` array whos values are the indices of the cluster for the
            corresponding element of `image`.

        `centers` (:class:`numpy.ndarray`):

            An `nclusters x B` array of cluster centers.

    Iterations are performed until clusters converge (no pixels reassigned
    between iterations), `max_iterations` is reached, or `compare` returns
    nonzero. If :exc:`KeyboardInterrupt` is generated (i.e., CTRL-C pressed)
    while the algorithm is executing, clusters are returned from the previously
    completed iteration.
    '''
    import spectral
    import numpy as np
    from spectral.algorithms.spymath import has_nan, NaNValueError

    if has_nan(image):
        raise NaNValueError('Image data contains NaN values.')

    status = spectral._status
    
    # defaults for kwargs
    start_clusters = None
    compare = None
    distance = L2
    iterations = None

    for (key, val) in list(kwargs.items()):
        if key == 'start_clusters':
            start_clusters = val
        elif key == 'compare':
            compare = val
        elif key == 'distance':
            if val in (L1, 'L1'):
                distance = L1
            elif val in (L2, 'L2'):
                distance = L2
            else:
                raise ValueError('Unrecognized keyword argument.')
        elif key == 'frames':
            if not hasattr(val, 'append'):
                raise TypeError('"frames" keyword argument must have "append"'
                                'attribute.')
            iterations = val
        else:
            raise NameError('Unsupported keyword argument.')

    (nrows, ncols, nbands) = image.shape
    N = nrows * ncols
    image = image.reshape((N, nbands))
    clusters = numpy.zeros((N,), int)
    if start_clusters is not None:
        assert (start_clusters.shape[0] == nclusters), 'There must be \
        nclusters clusters in the startCenters array.'
        centers = numpy.array(start_clusters)
    else:
        print('Initializing clusters along diagonal of N-dimensional bounding box.')
        boxMin = np.amin(image, 0)
        boxMax = np.amax(image, 0)
        delta = (boxMax - boxMin) / (nclusters - 1)
        centers = np.empty((nclusters, nbands), float)
        for i in range(nclusters):
            centers[i] = boxMin + i * delta

    distances = np.empty((N, nclusters), float)
    old_centers = np.array(centers)
    clusters = np.zeros((N,), int)
    old_clusters = np.copy(clusters)
    diffs = np.empty_like(image, dtype=np.float64)
    itnum = 1
    while (itnum <= max_iterations):
        try:
            status.display_percentage('Iteration %d...' % itnum)

            # Assign all pixels
            for i in range(nclusters):
                diffs = np.subtract(image, centers[i], out=diffs)
                if distance == L2:
                    distances[:, i] = np.einsum('ij,ij->i', diffs, diffs)
                else:
                    diffs = np.abs(diffs, out=diffs)
                    distances[:, i] = np.einsum('ij->i', diffs)
            clusters[:] = np.argmin(distances, 1)

            # Update cluster centers
            old_centers[:] = centers
            for i in range(nclusters):
                inds = np.argwhere(clusters == i)[:, 0]
                if len(inds) > 0:
                    centers[i] = np.mean(image[inds], 0, float)

            if iterations is not None:
                iterations.append(clusters.reshape(nrows, ncols))

            if compare and compare(old_clusters, clusters):
                status.end_percentage('done.')
                break
            else:
                nChanged = numpy.sum(clusters != old_clusters)
                if nChanged == 0:
                    status.end_percentage('0 pixels reassigned.')
                    break
                else:
                    status.end_percentage('%d pixels reassigned.' \
                                          % (nChanged))

            old_clusters[:] = clusters
            old_centers[:] = centers
            itnum += 1

        except KeyboardInterrupt:
            print("KeyboardInterrupt: Returning clusters from previous iteration.")
            return (old_clusters.reshape(nrows, ncols), old_centers)

    print('kmeans terminated with', len(set(old_clusters.ravel())), \
        'clusters after', itnum - 1, 'iterations.', file=status)
    return (old_clusters.reshape(nrows, ncols), centers)

