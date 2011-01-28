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

import numpy
from classifiers import Classifier

def L1(v1, v2):
    'Returns L1 distance between 2 rank-1 arrays.'
    return numpy.sum(abs((v1 - v2)))


def L2(v1, v2):
    'Returns Euclidean distance between 2 rank-1 arrays.'
    delta = v1 - v2
    return numpy.sqrt(numpy.dot(delta, delta))


class KmeansClusterer(Classifier):
    '''An unsupervised classifier using an iterative clustering algorithm'''
    def __init__(self, nClusters = 10, maxIter = 20, endCondition = None, distanceMeasure = L1):
        '''
        ARGUMENTS:
            nClusters       Number of clusters to create. Default is 8
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
        self.nClusters = nClusters
        self.maxIterations = maxIter
        self.endCondition = endCondition
        self.distanceMeasure = distanceMeasure
        
    def classifyImage(self, image, startClusters = None, iterations = None):
        '''
        Performs iterative self-organizing clustering of image data.

        USAGE: (clMap, centers) = cl.classifyImage(image
                                                   [, startClusters = None]
                                                   [, iterations = None])

        ARGUMENTS:
            image           A SpyFile or an MxNxB NumPy array
            startClusters   Initial cluster centers. This must be an
                            nClusters x B array.
            iterations      If this argument is passed and is a list object,
                            each intermediate cluster map is appended to
                            the list.
        RETURN VALUES:
            clMap           An MxN array whos values are the indices of the
                            cluster for the corresponding element of image.
            centers         An nClusters x B array of cluster centers.
        '''
        return isoCluster(image, self.nClusters, self.maxIterations, startClusters,
                          self.endCondition, self.distanceMeasure, iterations)
                
    
def kmeans(image, nClusters = 10, maxIterations = 20, **kwargs):
    '''
    Performs iterative clustering using the k-means algorithm.

    Arguments:
    
        `image` (:class:`numpy.ndarray` or :class:`spectral.Image`):
	
	    The `MxNxB` image on which to perform clustering.
	
        `nClusters` (int) [default 10]:
	
	    Number of clusters to create.  The number produced may be less than
	    `nClusters`.
	
        `maxIterations` (int) [default 20]:
	
	    Max number of iterations to perform.
    
    Keyword Arguments:
	
        `startClusters` (:class:`numpy.ndarray`) [default None]:
	
	    `nClusters x B` array of initial cluster centers.  If not provided,
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
    
        `clMap` (:class:`numpy.ndarray`):
	
	    An `MxN` array whos values are the indices of the cluster for the
	    corresponding element of `image`.
			
        `centers` (:class:`numpy.ndarray`):
	
	    An `nClusters x B` array of cluster centers.
    
    Iterations are performed until clusters converge (no pixels reassigned
    between iterations), `maxIterations` is reached, or `compare` returns nonzero.
    If :exc:`KeyboardInterrupt` is generated (i.e., CTRL-C pressed) while the
    algorithm is executing, clusters are returned from the previously completed
    iteration.
    '''
    from spectral import status
    import numpy
    
    if isinstance(image, numpy.ndarray):
	return kmeans_ndarray(*(image, nClusters, maxIterations), **kwargs)
    
    # defaults for kwargs
    startClusters = None
    compare = None
    distance = L2
    iterations = None
    
    for (key, val) in kwargs.items():
	if key == 'startClusters':
	    startClusters = val
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
		raise TypeError('"frames" keyword argument must have "append" attribute.')
	    iterations = frames
	else:
	    raise NameError('Unsupported keyword argument.')
	    
    (nRows, nCols, nBands) = image.shape
    clusters = numpy.zeros((nRows, nCols), int)
    oldClusters = numpy.copy(clusters)
    if startClusters != None:
        assert (startClusters.shape[0] == nClusters), 'There must be \
        nClusters clusters in the startCenters array.'
        centers = numpy.array(startClusters)
    else:
	print 'Initializing clusters along diagonal of N-dimensional bounding box.'
	centers = numpy.empty((nClusters, nBands), float)
	boxMin = image[0,0]
	boxMax = image[0,0]
	for i in range(nRows):
	    for j in range(nCols):
		x = image[i,j]
		boxMin = numpy.where(boxMin < x, boxMin, x)
		boxMax = numpy.where(boxMax > x, boxMax, x)
	boxMin = boxMin.astype(float)
	boxMax = boxMax.astype(float)
	delta = (boxMax - boxMin) / (nClusters - 1)
	for i in range(nClusters):
	    centers[i] = boxMin.astype(float) + i * delta

    print 'Starting iterations.'

    iter = 1
    while (iter <= maxIterations):
	try:
	    status.displayPercentage('Iteration %d...' % iter)
	    
	    # Assign all pixels
	    for i in range(nRows):
		status.updatePercentage(float(i) / nRows * 100.)
		for j in range(nCols):
		    minDist = 1.e30
		    for k in range(nClusters):
			dist = distance(image[i, j], centers[k])
			if (dist < minDist):
			    clusters[i, j] = k
			    minDist = dist
	    status.endPercentage()

	    # Update cluster centers
	    sums = numpy.zeros((nClusters, nBands), 'd')
	    counts = ([0] * nClusters)
	    for i in range(nRows):
		for j in range(nCols):
		    counts[clusters[i, j]] += 1
		    sums[clusters[i, j]] += image[i, j]
    
	    oldCenters = centers[:]
	    for i in range(nClusters):
		if (counts[i] > 0):
		    centers[i] = sums[i] / counts[i]
	    centers = numpy.array(centers)
    
	    if iterations != None:
		iterations.append(clusters)

	    if compare and compare(oldClusters, clusters):
		break
	    else:
		nChanged = numpy.sum(clusters != oldClusters)
		if nChanged == 0:
		    break
		else:
		    print >>status, '\t%d pixels reassigned.' % (nChanged)
    
	    oldClusters = clusters
	    oldCenters = centers
	    clusters = numpy.zeros((nRows, nCols), int)
	    iter += 1
	    
        except KeyboardInterrupt:
            print "KeyboardInterrupt: Returning clusters from previous iteration"
	    return (oldClusters, oldCenters)

    print >>status, 'kmeans terminated with', len(set(oldClusters.ravel())), \
          'clusters after', iter - 1, 'iterations.'
    return (oldClusters, centers)

def kmeans_ndarray(image, nClusters = 10, maxIterations = 20, **kwargs):
    '''
    Performs iterative clustering using the k-means algorithm.

    Arguments:
    
        `image` (:class:`numpy.ndarray` or :class:`spectral.Image`):
	
	    The `MxNxB` image on which to perform clustering.
	
        `nClusters` (int) [default 10]:
	
	    Number of clusters to create.  The number produced may be less than
	    `nClusters`.
	
        `maxIterations` (int) [default 20]:
	
	    Max number of iterations to perform.
    
    Keyword Arguments:
	
        `startClusters` (:class:`numpy.ndarray`) [default None]:
	
	    `nClusters x B` array of initial cluster centers.  If not provided,
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
    
        `clMap` (:class:`numpy.ndarray`):
	
	    An `MxN` array whos values are the indices of the cluster for the
	    corresponding element of `image`.
			
        `centers` (:class:`numpy.ndarray`):
	
	    An `nClusters x B` array of cluster centers.
    
    Iterations are performed until clusters converge (no pixels reassigned
    between iterations), `maxIterations` is reached, or `compare` returns nonzero.
    If :exc:`KeyboardInterrupt` is generated (i.e., CTRL-C pressed) while the
    algorithm is executing, clusters are returned from the previously completed
    iteration.
    '''
    from spectral import status
    import numpy as np
    
    # defaults for kwargs
    startClusters = None
    compare = None
    distance = L2
    iterations = None
    
    for (key, val) in kwargs.items():
	if key == 'startClusters':
	    startClusters = val
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
		raise TypeError('"frames" keyword argument must have "append" attribute.')
	    iterations = val
	else:
	    raise NameError('Unsupported keyword argument.')
	    
    (nRows, nCols, nBands) = image.shape
    N = nRows * nCols
    image = image.reshape((N, nBands))
    clusters = numpy.zeros((N,), int)
    if startClusters != None:
        assert (startClusters.shape[0] == nClusters), 'There must be \
        nClusters clusters in the startCenters array.'
        centers = numpy.array(startClusters)
    else:
	print 'Initializing clusters along diagonal of N-dimensional bounding box.'
	boxMin = np.amin(image, 0)
	boxMax = np.amax(image, 0)
	delta = (boxMax - boxMin) / (nClusters - 1)
	centers = np.empty((nClusters, nBands), float)
	for i in range(nClusters):
	    centers[i] = boxMin + i * delta

    print 'Starting iterations.'

    distances = np.empty((N, nClusters), float)
    oldCenters = np.array(centers)
    clusters = np.zeros((N,), int)
    oldClusters = np.copy(clusters)
    iter = 1
    while (iter <= maxIterations):
	try:
	    status.displayPercentage('Iteration %d...' % iter)

	    # Assign all pixels
	    for i in range(nClusters):
		diffs = np.subtract(image, centers[i])
		if distance == L2:
		    distances[:,i] = np.sum(np.square(diffs), 1)
		else:
		    distances[:,i] = np.sum(np.abs(diffs), 1)
	    clusters[:] = np.argmin(distances, 1)

	    status.endPercentage()

	    # Update cluster centers
	    oldCenters[:] = centers
	    for i in range(nClusters):
		inds = np.argwhere(clusters == i)[:,0]
		if len(inds) > 0:
		    centers[i] = np.mean(image[inds], 0, float)
		    
	    if iterations != None:
		iterations.append(clusters.reshape(nRows, nCols))

	    if compare and compare(oldClusters, clusters):
		break
	    else:
		nChanged = numpy.sum(clusters != oldClusters)
		if nChanged == 0:
		    break
		else:
		    print >>status, '\t%d pixels reassigned.' % (nChanged)
    
	    oldClusters[:] = clusters
	    oldCenters[:] = centers
	    iter += 1

        except KeyboardInterrupt:
            print "KeyboardInterrupt: Returning clusters from previous iteration."
	    return (oldClusters.reshape(nRows, nCols), oldCenters)

    print >>status, 'kmeans terminated with', len(set(oldClusters.ravel())), \
          'clusters after', iter - 1, 'iterations.'
    return (oldClusters.reshape(nRows, nCols), centers)

def isoCluster(*args, **kwargs):
    '''
    This is a deprecated function because it previously refered to a function that
    implemented the k-means clustering algorithm.  For backward compatibilty, this
    function will act as a pass through (with a warning issued) to the \"kmeans\"
    function, which is the correct name for the algorithm.  This \"isoCluster\"
    function will likely be dropped in a future version, unless an actual implementation
    of the ISO-Cluster algorithm is added.
    '''
    import warnings
    msg = "The function name \"isoCluster\" is deprecated since the function " \
	  "to which it refered is actually an implementation of the k-means " \
	  "clustering algorithm.  Please call \"kmeans\" instead."
    warnings.warn(msg, DeprecationWarning)
    return kmeans(*args, **kwargs)

def clusterOnePass(image, maxDist, nClusters = 10):
    '''
    A one-pass clustering algorithm.

    USAGE:  (clMap, centers) = clusterOnePass(image, maxDist
                                              [, nClusters = 10])
    ARGUMENTS:
        image           A SpyFile or an MxNxB NumPy array
        maxDist         The L1 distance at which a new cluster is created.
        nClusters       Number of clusters to create. Default is 10
    RETURN VALUES:
        clMap           An MxN array whos values are the indices of the
                        cluster for the corresponding element of image.
        centers         An nClusters x B array of cluster centers.

    The algorithm starts by using the pixel upperleft corner as the sole
    cluster center.  As each successive image pixel is compared to the
    set of cluster centers (we start with only one), if the smallest L1
    distance from the pixel to a cluster is greater than maxDist, the
    pixel is added to the set of clusters.

    Note that this algorithm is very sensitive to maxDist and can result
    in very many or very few (i.e., 1) clusters if maxDist is not chosen
    carefully.  For an alternate (better) one-pass clustering algorithm,
    see 'cluster'.
    '''
    import warnings
    warnings.warn('This function has been deprecated.')
    (nRows, nCols, nBands,) = (image.nRows, image.nCols, image.nBands)
    clusters = numpy.zeros((nRows, nCols), int)
    centers = [image[0, 0]]
    
    for i in range(nRows):
        for j in range(nCols):
            minDist = 1000000000000.0
            cluster = -1
            pixel = image[i, j]
            for k in range(len(centers)):
                dist = L1(pixel, centers[k])
                if (dist < minDist):
                    clusters[i, j] = k
                    minDist = dist

            if (minDist > maxDist):
                centers.append(pixel)

    return (clusters, centers)


class OnePassClusterer(Classifier):
    '''
    A class to implement a one-pass clustering algorithm with replacement.
    '''
    def __init__(self, maxClusters, maxDistance = 0, dist = L2):
        self.maxClusters = maxClusters
        self.maxDist = maxDistance
        self.dist = dist

    def addCluster(self, pixel):
        'Adds a new cluster center or replaces an existing one.'
        self.clusterMap = numpy.choose(numpy.equal(self.clusterMap, self.clusterToGo),
                                       (self.clusterMap, self.clusterToGoTo))
        self.clusters[self.clusterToGo] = pixel
        self.calcDistances()
        self.calcMinHalfDistances()
        self.calcMaxDistance()
        self.findNextToGo()

    def calcMinHalfDistances(self):
        '''
        For each cluster center, calculate half the distance to the
        nearest neighboring cluster.  This is used to quicken the
        cluster assignment loop.
        '''
        for i in range(self.nClusters):
            if (i == 0):
                self.minHalfDistance[i] = self.dist(self.clusters[0],
                                                    self.clusters[1])
            else:
                self.minHalfDistance[i] = self.dist(self.clusters[i],
                                                    self.clusters[0])
            for j in range(1, self.nClusters):
                if (j == i):
                    continue
                d = self.dist(self.clusters[i], self.clusters[j])
                if (d < self.minHalfDistance[i]):
                    self.minHalfDistance[i] = d

            self.minHalfDistance[i] *= 0.5

    def calcMinDistances(self):
        pass

    def calcDistances(self):
        self.distances = numpy.zeros((self.nClusters, self.nClusters), 'f')
        for i in range(self.nClusters):
            for j in range((i + 1), self.nClusters):
                self.distances[i, j] = self.dist(self.clusters[i],
                                                 self.clusters[j])
        self.distances += numpy.transpose(self.distances)

    def calcMaxDistance(self):
        'Determine greatest inter-cluster distance.'
        maxDist = 0
        for i in range(self.nClusters):
            rowMax = max(self.distances[i])
            if (rowMax > maxDist):
                maxDist = rowMax
        self.maxDist = maxDist

    def initClusters(self):
        'Assign initial cluster centers.'
        count = 0
        while (count < self.maxClusters):
            i = (count / self.image.shape[1])
            j = (count % self.image.shape[1])
            self.clusters[count] = self.image[i, j]
            count += 1

        self.nClusters = self.maxClusters
        self.minHalfDistance = numpy.zeros(self.nClusters, float)
        self.calcMinHalfDistances()

	########################################################################
	# TO DO:  Need to see if there is a good reason why we're not just
	# calling self.calcDistances instead of using the code below.
	########################################################################

        self.distances = numpy.zeros((self.nClusters, self.nClusters), numpy.float)
        for i in range((self.nClusters - 1)):
            for j in range((i + 1), self.nClusters):
                self.distances[i, j] = self.dist(self.clusters[i],
                                                 self.clusters[j])
        self.findNextToGo()

    def findNextToGo(self):
        '''
        Determine which cluster is the next to be consolidated and
        into which other cluster it will go.  First, determine which 2
        clusters are closest and eliminate the one of those 2 which is
        nearest to any other cluster.
        '''
        from numpy import argmin, argsort, zeros
        mins = zeros(self.nClusters)
        for i in range(self.nClusters):
            mins[i] = numpy.argsort(self.distances[i])[1]

        a = argmin(mins)
        b = argsort(self.distances[a])[1]

	if self.nClusters > 2:
	    aNext = argsort(self.distances[a])[2]
	    bNext = argsort(self.distances[b])[2]
	    aNextDist = self.dist(self.clusters[a], self.clusters[aNext])
	    bNextDist = self.dist(self.clusters[b], self.clusters[bNext])
	    if (aNextDist > bNextDist):
		self.clusterToGo = b
		self.clusterToGoTo = a
	    else:
		self.clusterToGo = a
		self.clusterToGoTo = b
	else:
	    self.clusterToGo = a
	    self.clusterToGoTo = b

    def classifyImage(self, image):
        from spectral import status
        from spectral.io.spyfile import SpyFile
        self.image = image
        self.clusterMap = numpy.zeros(self.image.shape[:2], int)
        if isinstance(image, SpyFile):
            typecode = image.typecode()
        else:
            typecode = 'f'
        self.clusters = numpy.zeros((self.maxClusters, self.image.shape[2]), typecode)
        self.nClusters = 0
        clusters = self.clusters
        self.initClusters()
        hd = 0
        status.displayPercentage('Clustering image...')
        for i in range(image.shape[0]):
            status.updatePercentage(float(i) / image.shape[0] * 100.)
            for j in range(image.shape[1]):
                minDistance = self.maxDist
                for k in range(len(clusters)):
                    d = self.dist(image[i, j], clusters[k])
                    if (d < minDistance):
                        self.clusterMap[i, j] = k
                        minDistance = d

                if (minDistance == self.maxDist):
                    cl = self.clusterToGo
                    self.addCluster(image[i, j])
                    self.clusterMap[i, j] = cl
        status.endPercentage()
        self.image = None
        return (self.clusterMap, self.clusters)


def cluster(data, nClusters = 10):
    '''
    An single-pass clustering algorithm with replacement.

    Arguments:

        `data` (:class:`numpy.ndarray` or :class:`spectral.Image`):
	
	    The `MxNxB` image on which to perform clustering.

        `nClusters` (int) [default 10]:
	
	    Number of clusters to create.

    Returns a 2-tuple containing:

        `clMap` (:class:`numpy.ndarray`):
	
	    An `MxN` array of cluster indices.

        `centers` (:class:`numpy.ndarray`):
	
	    A `nClusters x B` array of cluster centers corresponding to the
	    indices in clMap

    This algorithm initializes the clusters with the first `nClusters`
    pixels from data.  Successive pixels are then assigned to the nearest
    cluster in `N`-space.  If the distance from a pixel to the nearest cluster
    is greater than the greatest inter-cluster distance, the pixel is added
    as a new cluster and the two clusters nearest to eachother are combined into
    a single cluster.

    The advantages of this algorithm are that threshold distances need
    not be specified and the number of clusters remains fixed; however, results
    typically are not as accurate as iterative algorithms.
    '''
    opc = OnePassClusterer(nClusters)
    return opc.classifyImage(data)


