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


def isoCluster(*args, **kwargs):
    '''
    This is a deprecated function because it previously refered to a function
    that implemented the k-means clustering algorithm.  For backward
    compatibilty, this function will act as a pass through (with a warning
    issued) to the \"kmeans\" function, which is the correct name for the
    algorithm.  This \"isoCluster\" function will likely be dropped in a future
    version, unless an actual implementation of the ISO-Cluster algorithm is
    added.
    '''
    import warnings
    msg = "The function name \"isoCluster\" is deprecated since the function " \
          "to which it refered is actually an implementation of the k-means " \
          "clustering algorithm.  Please call \"kmeans\" instead."
    warnings.warn(msg, DeprecationWarning)
    return kmeans(*args, **kwargs)


def clusterOnePass(image, max_dist, nclusters=10):
    '''
    A one-pass clustering algorithm.

    USAGE:  (clMap, centers) = clusterOnePass(image, max_dist
                                              [, nclusters = 10])
    ARGUMENTS:
        image           A SpyFile or an MxNxB NumPy array
        max_dist         The L1 distance at which a new cluster is created.
        nclusters       Number of clusters to create. Default is 10
    RETURN VALUES:
        clMap           An MxN array whos values are the indices of the
                        cluster for the corresponding element of image.
        centers         An nclusters x B array of cluster centers.

    The algorithm starts by using the pixel upperleft corner as the sole
    cluster center.  As each successive image pixel is compared to the
    set of cluster centers (we start with only one), if the smallest L1
    distance from the pixel to a cluster is greater than max_dist, the
    pixel is added to the set of clusters.

    Note that this algorithm is very sensitive to max_dist and can result
    in very many or very few (i.e., 1) clusters if max_dist is not chosen
    carefully.  For an alternate (better) one-pass clustering algorithm,
    see 'cluster'.
    '''
    import warnings
    warnings.warn('This function has been deprecated.', DeprecationWarning)
    (nrows, ncols, nbands,) = (image.nrows, image.ncols, image.nbands)
    clusters = numpy.zeros((nrows, ncols), int)
    centers = [image[0, 0]]

    for i in range(nrows):
        for j in range(ncols):
            minDist = 1000000000000.0
            cluster = -1
            pixel = image[i, j]
            for k in range(len(centers)):
                dist = L1(pixel, centers[k])
                if (dist < minDist):
                    clusters[i, j] = k
                    minDist = dist

            if (minDist > max_dist):
                centers.append(pixel)

    return (clusters, centers)


class OnePassClusterer(Classifier):
    '''
    A class to implement a one-pass clustering algorithm with replacement.
    '''
    def __init__(self, max_clusters, max_distance=0, dist=L2):
        self.max_clusters = max_clusters
        self.max_dist = max_distance
        self.dist = dist

    def add_cluster(self, pixel):
        'Adds a new cluster center or replaces an existing one.'
        self.cluster_map = numpy.choose(
            numpy.equal(self.cluster_map, self.cluster_to_go),
            (self.cluster_map, self.cluster_to_go_to))
        self.clusters[self.cluster_to_go] = pixel
        self.calc_distances()
        self.calc_min_half_distances()
        self.calc_max_distance()
        self.find_next_to_go()

    def calc_min_half_distances(self):
        '''
        For each cluster center, calculate half the distance to the
        nearest neighboring cluster.  This is used to quicken the
        cluster assignment loop.
        '''
        for i in range(self.nclusters):
            if (i == 0):
                self.min_half_dist[i] = self.dist(self.clusters[0],
                                                  self.clusters[1])
            else:
                self.min_half_dist[i] = self.dist(self.clusters[i],
                                                  self.clusters[0])
            for j in range(1, self.nclusters):
                if (j == i):
                    continue
                d = self.dist(self.clusters[i], self.clusters[j])
                if (d < self.min_half_dist[i]):
                    self.min_half_dist[i] = d

            self.min_half_dist[i] *= 0.5

    def calcMinDistances(self):
        pass

    def calc_distances(self):
        self.distances = numpy.zeros((self.nclusters, self.nclusters), 'f')
        for i in range(self.nclusters):
            for j in range((i + 1), self.nclusters):
                self.distances[i, j] = self.dist(self.clusters[i],
                                                 self.clusters[j])
        self.distances += numpy.transpose(self.distances)

    def calc_max_distance(self):
        'Determine greatest inter-cluster distance.'
        max_dist = 0
        for i in range(self.nclusters):
            rowMax = max(self.distances[i])
            if (rowMax > max_dist):
                max_dist = rowMax
        self.max_dist = max_dist

    def init_clusters(self):
        'Assign initial cluster centers.'
        count = 0
        while (count < self.max_clusters):
            i = (count / self.image.shape[1])
            j = (count % self.image.shape[1])
            self.clusters[count] = self.image[i, j]
            count += 1

        self.nclusters = self.max_clusters
        self.min_half_dist = numpy.zeros(self.nclusters, float)
        self.calc_min_half_distances()

        #######################################################################
        # TO DO:  Need to see if there is a good reason why we're not just
        # calling self.calcDistances instead of using the code below.
        #######################################################################

        self.distances = numpy.zeros(
            (self.nclusters, self.nclusters), numpy.float)
        for i in range((self.nclusters - 1)):
            for j in range((i + 1), self.nclusters):
                self.distances[i, j] = self.dist(self.clusters[i],
                                                 self.clusters[j])
        self.find_next_to_go()

    def find_next_to_go(self):
        '''
        Determine which cluster is the next to be consolidated and
        into which other cluster it will go.  First, determine which 2
        clusters are closest and eliminate the one of those 2 which is
        nearest to any other cluster.
        '''
        from numpy import argmin, argsort, zeros
        mins = zeros(self.nclusters)
        for i in range(self.nclusters):
            mins[i] = numpy.argsort(self.distances[i])[1]

        a = argmin(mins)
        b = argsort(self.distances[a])[1]

        if self.nclusters > 2:
            aNext = argsort(self.distances[a])[2]
            bNext = argsort(self.distances[b])[2]
            aNextDist = self.dist(self.clusters[a], self.clusters[aNext])
            bNextDist = self.dist(self.clusters[b], self.clusters[bNext])
            if (aNextDist > bNextDist):
                self.cluster_to_go = b
                self.cluster_to_go_to = a
            else:
                self.cluster_to_go = a
                self.cluster_to_go_to = b
        else:
            self.cluster_to_go = a
            self.cluster_to_go_to = b

    def classify_image(self, image):
        import spectral
        from spectral.io.spyfile import SpyFile
        status = spectral._status
        self.image = image
        self.cluster_map = numpy.zeros(self.image.shape[:2], int)
        self.clusters = numpy.zeros((self.max_clusters, self.image.shape[2]),
                                    self.image.dtype)
        self.nclusters = 0
        clusters = self.clusters
        self.init_clusters()
        hd = 0
        status.display_percentage('Clustering image...')
        for i in range(image.shape[0]):
            status.update_percentage(float(i) / image.shape[0] * 100.)
            for j in range(image.shape[1]):
                minDistance = self.max_dist
                for k in range(len(clusters)):
                    d = self.dist(image[i, j], clusters[k])
                    if (d < minDistance):
                        self.cluster_map[i, j] = k
                        minDistance = d

                if (minDistance == self.max_dist):
                    cl = self.cluster_to_go
                    self.add_cluster(image[i, j])
                    self.cluster_map[i, j] = cl
        status.end_percentage()
        self.image = None
        return (self.cluster_map, self.clusters)


def cluster(data, nclusters=10):
    '''
    An single-pass clustering algorithm with replacement.

    Arguments:

        `data` (:class:`numpy.ndarray` or :class:`spectral.Image`):

            The `MxNxB` image on which to perform clustering.

        `nclusters` (int) [default 10]:

            Number of clusters to create.

    Returns a 2-tuple containing:

        `class_map` (:class:`numpy.ndarray`):

            An `MxN` array of cluster indices.

        `centers` (:class:`numpy.ndarray`):

            A `nclusters x B` array of cluster centers corresponding to the
            indices in clMap

    This algorithm initializes the clusters with the first `nclusters`
    pixels from data.  Successive pixels are then assigned to the nearest
    cluster in `N`-space.  If the distance from a pixel to the nearest cluster
    is greater than the greatest inter-cluster distance, the pixel is added
    as a new cluster and the two clusters nearest to eachother are combined
    into a single cluster.

    The advantages of this algorithm are that threshold distances need
    not be specified and the number of clusters remains fixed; however, results
    typically are not as accurate as iterative algorithms.
    '''
    opc = OnePassClusterer(nclusters)
    return opc.classify_image(data)
