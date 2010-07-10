#########################################################################
#
#   Cluster.py - This file is part of the Spectral Python (SPy) package.
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
Unsupervised clustering algorithms.
'''

import numpy
from Classifiers import Classifier

def L1(v1, v2):
    'Returns L1 distance between 2 rank-1 arrays.'
    return numpy.sum(abs((v1 - v2)))


def L2(v1, v2):
    'Returns Euclidean distance between 2 rank-1 arrays.'
    delta = numpy.array(v1 - v2, float)
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
                
    
def kmeans(image, nClusters = 10, maxIter = 20, startClusters = None,
           compare = None, distance = L1, iterations = None):
    '''
    Performs iterative clustering using the k-means algorithm.

    USAGE: (clMap, centers) = kmeans(image [, nClusters = 8]
                                           [, maxIter = 20]
                                           [, startClusters = None]
                                           [, compare = None]
                                           [, distance = L1]
                                           [, iterations = None])

    ARGUMENTS:
        image           A SpyFile or an MxNxB NumPy array
        nClusters       Number of clusters to create. Default is 8
        maxIter         Max number of iterations. Default is 20
        startClusters   Initial cluster centers. This must be an
                        nClusters x B array.
        compare         Optional comparison function. compare must be a
                        function which takes 2 MxN NumPy arrays as its
                        arguments and returns non-zero when clustering
                        is to be terminated. The two arguments are the
                        cluster maps for the previous and current cluster
                        cycle, respectively.
        distance        The distance measure to use for comparison. The
                        default is the L1 distance. For  Euclidean
                        distance, specify L2 (no quotes).
        iterations      If this argument is passed and is a list object,
                        each intermediate cluster map is appended to
                        the list.
    RETURN VALUES:
        clMap           An MxN array whos values are the indices of the
                        cluster for the corresponding element of image.
        centers         An nClusters x B array of cluster centers.
    '''
    from Spectral import status
    if not isinstance(iterations, list):
        iterations = None
    (nRows, nCols, nBands) = image.shape
    clusters = numpy.zeros((nRows, nCols), int)
    oldClusters = None
    if startClusters != None:
        assert (startClusters.shape[0] == nClusters), 'There must be \
        nClusters clusters in the startCenters array.'
        centers = numpy.array(startClusters)
    else:
        maxVal = 5000.0
        centers = []
        for i in range(nClusters):
            centers.append((((numpy.ones(nBands) * i) * maxVal) / nClusters))
        centers = numpy.array(centers)

    iter = 1
    while (iter <= maxIter):
        status.displayPercentage('Iteration %d...' % iter)
        for i in range(nRows):
            status.updatePercentage(float(i) / nRows * 100.)
            for j in range(nCols):
                minDist = 10000000000000.0
                for k in range(len(centers)):
                    dist = distance(image[i, j], centers[k])
                    if (dist < minDist):
                        clusters[i, j] = k
                        minDist = dist
        status.endPercentage()

        sums = numpy.zeros((nClusters, nBands), float)
        counts = ([0] * nClusters)
        for i in range(nRows):
            for j in range(nCols):
                counts[clusters[i, j]] += 1
                sums[clusters[i, j]] += image[i, j]


        centers = []
        for i in range(nClusters):
            if (counts[i] > 0):
                centers.append((sums[i] / counts[i]))
        centers = numpy.array(centers)

        nClusters = centers.shape[0]
        if compare:
            if compare(oldClusters, clusters):
                print >>status, '\tisoCluster converged with', centers.shape[0], \
                      'clusters in', iter, 'iterations.'
                return (clusters, centers)
        elif oldClusters != None:
            nChanged = abs(sum((clusters - oldClusters).ravel()))
            if nChanged == 0:
                print >>status, '\tisoCluster converged with', centers.shape[0], \
                      'clusters in', iter, 'iterations.'
                return (clusters, centers)
            else:
                print >>status, '\t%d pixels reassigned.' % (nChanged)

        oldClusters = clusters
        if iterations != None:
            iterations.append(oldClusters)
        clusters = numpy.zeros((nRows, nCols), int)
        iter += 1

    print >>status, '\kmeans terminated with', centers.shape[0], \
          'clusters after', iter - 1, 'iterations.'
    return (oldClusters, centers)

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
        self.distances = numpy.zeros((self.nClusters, self.nClusters))
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

    def classifyImage(self, image):
        from Spectral import status
        from Spectral.Io.SpyFile import SpyFile
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
    An efficient one-pass clustering algorithm with replacement.

    USAGE: (clMap, centers) = cluster(data [, nClusters = 10])

    ARGUMENTS:
        data            A SpyFile or an MxNxB array
        nClusters       Optional number of clusters to Create.
                        The default is 10.
    RETURN VALUES:
        clMap           an MxN array of cluster indices
        centers         An MxB array of cluster centers corresponding to
                        the indices in clMap

    This algorithm initializes the clusters with the first nClusters
    pixels from data.  Successive pixels are then assigned to the nearest
    cluster.  If the distance from a pixel to the nearest cluster is
    greater than the greatest inter-cluster distance, the pixel is added
    as a new cluster and the two clusters nearest to eachother are
    combined into a single cluster.

    The advantages of this algorithm are that threshold distances need
    not be specified and the number of clusters remains fixed.
    '''
    opc = OnePassClusterer(nClusters)
    return opc.classifyImage(data)


