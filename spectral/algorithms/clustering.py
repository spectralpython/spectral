'''
k-means clustering.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np

import spectral as spy
from ..utilities.errors import has_nan, NaNValueError


def L1(v1, v2):
    'Returns L1 distance between 2 rank-1 arrays.'
    return np.sum(abs((v1 - v2)))


def L2(v1, v2):
    'Returns Euclidean distance between 2 rank-1 arrays.'
    delta = v1 - v2
    return np.sqrt(np.dot(delta, delta))


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

            An `MxN` array who's values are the indices of the cluster for the
            corresponding element of `image`.

        `centers` (:class:`numpy.ndarray`):

            An `nclusters x B` array of cluster centers.

    Iterations are performed until clusters converge (no pixels reassigned
    between iterations), `maxIterations` is reached, or `compare` returns
    nonzero. If :exc:`KeyboardInterrupt` is generated (i.e., CTRL-C pressed)
    while the algorithm is executing, clusters are returned from the previously
    completed iteration.
    '''
    logger = logging.getLogger('spectral')

    if isinstance(image, np.ndarray):
        return kmeans_ndarray(*(image, nclusters, max_iterations), **kwargs)

    status = spy._status

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
    clusters = np.zeros((nrows, ncols), int)
    old_clusters = np.copy(clusters)
    if start_clusters is not None:
        assert (start_clusters.shape[0] == nclusters), 'There must be \
        nclusters clusters in the startCenters array.'
        centers = np.array(start_clusters)
    else:
        logging.debug('Initializing clusters along diagonal of N-dimensional bounding box.')
        centers = np.empty((nclusters, nbands), float)
        boxMin = image[0, 0]
        boxMax = image[0, 0]
        for i in range(nrows):
            for j in range(ncols):
                x = image[i, j]
                boxMin = np.where(boxMin < x, boxMin, x)
                boxMax = np.where(boxMax > x, boxMax, x)
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
            sums = np.zeros((nclusters, nbands), 'd')
            counts = ([0] * nclusters)
            for i in range(nrows):
                for j in range(ncols):
                    counts[clusters[i, j]] += 1
                    sums[clusters[i, j]] += image[i, j]

            old_centers = centers[:]
            for i in range(nclusters):
                if (counts[i] > 0):
                    centers[i] = sums[i] / counts[i]
            centers = np.array(centers)

            if iterations is not None:
                iterations.append(clusters)

            if compare and compare(old_clusters, clusters):
                status.end_percentage('done.')
                break
            else:
                nChanged = np.sum(clusters != old_clusters)
                if nChanged == 0:
                    status.end_percentage('0 pixels reassigned.')
                    break
                else:
                    status.end_percentage('%d pixels reassigned.' \
                                          % (nChanged))

            old_clusters = clusters
            old_centers = centers
            clusters = np.zeros((nrows, ncols), int)
            itnum += 1

        except KeyboardInterrupt:
            print("KeyboardInterrupt: Returning clusters from previous iteration")
            return (old_clusters, old_centers)

    logger.info('kmeans terminated with %d clusters after %d iterations',
                len(set(old_clusters.ravel())), itnum - 1)
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

            An `MxN` array who's values are the indices of the cluster for the
            corresponding element of `image`.

        `centers` (:class:`numpy.ndarray`):

            An `nclusters x B` array of cluster centers.

    Iterations are performed until clusters converge (no pixels reassigned
    between iterations), `max_iterations` is reached, or `compare` returns
    nonzero. If :exc:`KeyboardInterrupt` is generated (i.e., CTRL-C pressed)
    while the algorithm is executing, clusters are returned from the previously
    completed iteration.
    '''
    logger = logging.getLogger('spectral')

    if has_nan(image):
        raise NaNValueError('Image data contains NaN values.')

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
    clusters = np.zeros((N,), int)
    if start_clusters is not None:
        assert (start_clusters.shape[0] == nclusters), 'There must be \
        nclusters clusters in the startCenters array.'
        centers = np.array(start_clusters)
    else:
        logger.debug('Initializing clusters along diagonal of N-dimensional' \
                     ' bounding box.')
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
                break
            else:
                nChanged = np.sum(clusters != old_clusters)
                logger.info('k-means iteration {} - {} pixels reassigned.' \
                            .format(itnum, nChanged))
                if nChanged == 0:
                    break

            old_clusters[:] = clusters
            old_centers[:] = centers
            itnum += 1

        except KeyboardInterrupt:
            print("KeyboardInterrupt: Returning clusters from previous iteration.")
            return (old_clusters.reshape(nrows, ncols), old_centers)

    logger.info('kmeans terminated with %d clusters after %d iterations.',
                len(set(old_clusters.ravel())), itnum - 1)
    return (old_clusters.reshape(nrows, ncols), centers)
