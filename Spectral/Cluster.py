#########################################################################
#
#   Cluster.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2001 Thomas Boggs
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

from Numeric import *

def L1(v1, v2):
    'Returns Euclidean distance between 2 rank-1 arrays.'
    return sum(abs((v1 - v2)))


def L2(v1, v2):
    'Returns Euclidean distance between 2 rank-1 arrays.'
    delta = (v1 - v2)
    return dot(delta, delta)


def kMeans(image, nClusters = 8, maxIter = 20, compare = None):
    '''
    Performs K-means clustering on image data.

    USAGE: (clMap, centers) = kMeans(image [, nClusters = 8]
                                     [, maxIter = 20]
                                     [, startCenters = None]
                                     [, compare = None]
                                     [, distance = L1])

    ARGUMENTS:
        image           A SpyFile or an MxNxB NumPy array
        nClusters       Number of clusters to create. Default is 8
        maxIter         Max number of iterations. Default is 20
        startCenters    Initial cluster centers. This must be an
                        nClusters x B array.
        compare         Optional comparison funtion. compare must be a
                        function which takes 2 MxN NumPy arrays as its
                        arguments and returns non-zero when clustering
                        is to be terminated. The two arguments are the
                        cluster maps for the previous and current cluster
                        cycle, respectively.
        distance        The distance measure to use for comparison. The
                        default is the L1 distance. For  Euclidean
                        distance, specify L2 (no quotes).
    RETURN VALUES:
        clMap           An MxN array whos values are the indices of the
                        cluster for the corresponding element of image.
        centers         An nClusters x B array of cluster centers.
    '''
    (nRows, nCols, nBands,) = (image.nRows, image.nCols, image.nBands)
    clusters = zeros((nRows, nCols))
    oldClusters = None
    if startClusters:
        assert (startClusters.shape[1] == nClusters), 'There must be \
        nClusters clusters in the startCenters array.'
        centers = array(startClusters)
    else:
        maxVal = 5000.0
        centers = []
        for i in range(nClusters):
            centers.append((((ones(nBands) * i) * maxVal) / nClusters))

    iter = 0
    while (iter < maxIter):
        for i in range(nRows):
            for j in range(nCols):
                minDist = 10000000000000.0
                center = 0
                for k in range(len(centers)):
                    dist = sum(abs((image[i, j] - centers[k])))
                    if (dist < minDist):
                        clusters[i, j] = k
                        minDist = dist

        sums = zeros((nClusters, nBands), Float)
        counts = ([0] * nClusters)
        for i in range(nRows):
            for j in range(nCols):
                counts[clusters[i, j]] += 1
                sums[clusters[i, j]] += image[i, j]


        centers = []
        for i in range(nClusters):
            if (counts[i] > 0):
                print i
                centers.append((sums[i] / counts[i]))

        nClusters = len(centers)
        if compare:
            if compare(oldClusters, clusters):
                return (clusters, centers)
        elif (oldClusters and (sum(sum((clusters - oldClusters))) == 0)):
            return (clusters, centers)
        else:
            oldClusters = clusters
            clusters = zeros((nRows, nCols))
        iter += 1

    print 'kMeans terminated after', iter, 'iterations.'
    return (clusters, centers)


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
    distance from the pixel to a cluster is greater then maxDist, the
    pixel is added to the set of clusters.

    Note that this algorithm is very sensitive to maxDist and can result
    in very many or very few (i.e., 1) clusters if maxDist is not chosen
    carefully.  For an alternate one-pass clustering algorithm, see
    'cluster'.
    '''
    from NumTut import view
    (nRows, nCols, nBands,) = (image.nRows, image.nCols, image.nBands)
    clusters = zeros((nRows, nCols))
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


class OnePassClusterEngine:
    '''
    A class to implement a one-pass clustering algorithm with replacement.
    '''
    from Numeric import *

    def __init__(self, image, maxClusters, maxDistance = 0, dist = L2):
        self.image = image
        self.maxClusters = maxClusters
        self.maxDist = maxDistance
        self.dist = dist
        self.clusterMap = zeros(self.image.shape[:2], Int)
        self.clusters = zeros((self.maxClusters, self.image.shape[2]))
        self.nClusters = 0


    def addCluster(self, pixel):
        'Adds a new cluster center or replaces an existing one.'
        self.clusterMap = choose(equal(self.clusterMap, self.clusterToGo),
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
        self.distances = zeros((self.nClusters, self.nClusters))
        for i in range(self.nClusters):
            for j in range((i + 1), self.nClusters):
                self.distances[i, j] = self.dist(self.clusters[i],
                                                 self.clusters[j])
        self.distances += transpose(self.distances)


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
        self.minHalfDistance = zeros(self.nClusters, Float)
        self.calcMinHalfDistances()
        self.distances = zeros((self.nClusters, self.nClusters), Float)
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
        mins = zeros(self.nClusters)
        for i in range(self.nClusters):
            mins[i] = argsort(self.distances[i])[1]

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

    def go(self):
        image = self.image
        clusters = self.clusters
        self.initClusters()
        hd = 0
        for i in range(image.shape[0]):
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
    e = OnePassClusterEngine(data, nClusters)
    e.go()
    return (e.clusterMap, e.clusters)


