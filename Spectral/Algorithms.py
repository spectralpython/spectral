#########################################################################
#
#   Algorithms.py - This file is part of the Spectral Python (SPy)
#   package.
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

'''
Various functions and algorithms for processing spectral data.
'''

from Numeric import *
from LinearAlgebra import *


def mean_cov(vectors):
    '''
    Return the mean and covariance of the set of vectors.

    USAGE: (mean, cov) = mean_cov(vectors)

    ARGUMENTS:
        vectors         A SpyFile object or an MxNxB array
    RETURN VALUES:
        mean            The mean value of the vectors
        cov             The unbiased estimate (dividing by N-1) of
                        the covariance of the vectors.

    Calculate the mean and covariance of of the given vectors. The argument
    is either M vectors of length N in an M x N array, an M x N array of
    length B vectors, or a SpyFile object.
    '''

    from Numeric import *
    import time
    
    if len(vectors.shape) == 3 and vectors.shape[2] > 1:
    	#  Assuming vectors are along 3rd dimension
	(M, N, B) = vectors.shape
	mean = zeros((B,), Float)
	cov = zeros((B, B), Float)
	for i in range(M):
            print '\tMean: %5.1f%%' % (float(i) / M * 100.)
	    for j in range(N):
	    	mean += vectors[i, j]
	mean /= float(M * N)
	
	for i in range(M):
            print '\tCovariance: %5.1f%%' % (float(i) / M * 100.)
	    for j in range(N):
		x = (vectors[i, j] - mean)[:, NewAxis].astype(Float)
		cov += matrixmultiply(x, transpose(x))
	cov /= float(M * N - 1)
	return (mean, cov)
	
    else:
    	# Assuming vectors are in columns in 2D array	    
	(M, N) = vectors.shape[:2]
	mean = array(sum(vectors, 0)).astype(Float) / M
	cov = zeros((N, N), Float)

	for i in range(M):
            print '\tCovariance: %5.1f%%' % (float(i) / M * 100.)
	    x = (vectors[i] - mean)[:, NewAxis]
	    cov += matrixmultiply(x, transpose(x))
	cov /= (M - 1)
    	return (mean, cov)


def principalComponents(image):
    '''
    Calculate Principal Component eigenvalues & eigenvectors 
    for an image.

    USAGE:  (L, V, M, C) = principalComponents(image)

    ARGUMENTS:\n
        image -          A SpyFile object or an MxNxB array\n
    RETURN VALUES:\n
        L      -         A length B array of eigenvalues\n
        V      -         A BxB array of normalized eigenvectors\n
        M      -         The length B mean of the image pixels\n
        C      -         The BxB covariance matrix of the image\n
    '''
    
    from LinearAlgebra import eigenvectors
    
    (M, N, B) = image.shape
    
    (mean, cov) = mean_cov(image)
    (L, V) = eigenvectors(cov)

    #  Normalize eigenvectors
    V = V / sqrt(sum(V * V))

    return (L, V, mean, cov)



def canonicalAnalysis(classes):
    '''
    Solve for canonical eigenvalues and eigenvectors.

    USAGE: (L, V, CB, CW) = canonicalAnalysis(classes)

    Determines the solution to the generalized eigenvalue problem
    
            cov_b * x = lambda * cov_w * x
            
    Since cov_w is normally invertable, the reduces to
    
            (inv(cov_w) * cov_b) * x = lambda * x
            
    The return value is a 4-tuple containing the vector of eigenvalues,
    a matrix of the corresponding eigenvectors, the between-class
    covariance matrix, and the within-class covariance matrix.
    '''

    from LinearAlgebra import inverse, eigenvectors
    import math

    B = classes[0].cov.shape[0]	# Number of bands
    C = len(classes)		# Number of training sets
    rank = len(classes) - 1

    # Calculate total # of training pixels and total mean
    N = 0
    mean = zeros(B, Float)
    for s in classes: 
	N += s.size()
	mean += s.size() * s.mean
    mean /= float(N)

    cov_b = zeros((B, B), Float)
    cov_w = zeros((B, B), Float)

    for s in classes: 
	cov_w += (s.size() - 1) * s.cov
	m = (s.mean - mean)[:, NewAxis]
	cov_b += s.size() * matrixmultiply(m, transpose(m))
    cov_w /= float(N)
    cov_b /= float(N)
    
    cwInv = inverse(cov_w)
    (vals, vecs) = eigenvectors(matrixmultiply(cwInv, cov_b))

    vals = vals[:rank]
    vecs = vecs[:rank, :]

    # Diagonalize cov_within in the new space
    v = matrixmultiply(vecs, matrixmultiply(cov_w, transpose(vecs)))
    d = diagonal(v)
#    vecs /= sqrt(d) #[:, NewAxis]
    for i in range(vecs.shape[0]):
    	vecs[i, :] /= math.sqrt(d[i].real)
    	
    return (vals.real, vecs.real, cov_b, cov_w)



def reduceEigenvectors(L, V, fraction = 0.99):
    '''
    Reduces number of eigenvalues and eigenvectors retained.

    USAGE: (L2, V2) = reduceEigenvectors(L, V [, fraction])

    ARGUMENTS:
        L               A vector of descending eigenvalues
        V               The array of eigenvectors corresponding to L
        fraction        The fraction of sum(L) to retain
    RETURN VALUES:
        L2              A vector containing the first N eigenvalues of
                        L such that sum(L2) / sum(L) >= fraction
        V2              The array of eigenvectors corresponding to L2

    Retains only the first N eigenvalues and eigenvectors such that the
    sum of the retained eigenvalues divided by the sum of all eigenvalues
    is greater than or equal to fraction.  If fraction is not specified,
    the default value of 0.99 is used.
    '''

    import Numeric

    cumEig = Numeric.cumsum(L)
    sum = cumEig[-1]
    # Count how many values to retain.
    for i in range(len(L)):
	if (cumEig[i] / sum) >= fraction:
	    break

    if i == (len(L) - 1):
	# No reduction
	return (L, V)

    # Return cropped eigenvalues and eigenvectors
    L = L[:i + 1]
    V = V[:i + 1, :]
    return (L, V)

def logDeterminant(x):
    return sum(log(eigenvalues(x)))

class TrainingSet:
    def __init__(self, image, mask, index = 0, classProb = 1.0):
        '''
        Define a Training Set by selecting a set of vectors from image
        by applying the given mask.

        USAGE: ts = TrainingSet(image, mask [, index] [classProb = 1.0)

        image specifies the data source for which the training set is
        being defined. If index is not specified, the training set is
        defined by all elements of image for which the corresponding
        element if mask is non-zero. If index is non-zero, the training
        set is defined by all elements of image for which mask equals
        index. classProb represents the fractional abundance of the
        spectral class in the entire image. If this value is not
        specified, the default of 1 is used (Note that if all training
        sets use a fractional abundance of 1, they will all have equal
        weighting during classification).
        '''
        self.image = image
        self.mask = mask
        self.index = index
        self.classProb = classProb

        self._statsValid = 0
        self._size = 0

    def statsValid(self, tf):
        '''
        Set statistics for the TrainingSet to be valid or invalid.

        USAGE: tset.statsValid(bool)

        This function is intended to be called with a zero argument
        when the set's statistics become invalid (e.g., the data
        source or mask changes).
        '''
        self._statsValid = tf

    def size(self):
        '''Return the number of pixels in the training set.'''

        # If the stats are invalid, the number of pixels in the
        # training set may have changed.
        if self._statsValid:
            return self._size

        if self.index:
            return sum(equal(self.mask, self.index).flat)
        else:
            return sum(not_equal(self.mask, 0).flat)
        

    def calcStatistics(self):
        '''
        Calculates statistic for the class.
        '''

        from LinearAlgebra import *

        (nRows, nCols, nBands) = self.image.shape

        # Get the proper mask for the training set
        if self.index:
            sMask = equal(self.mask, self.index)
        else:
            sMask = not_equal(self.mask, 0)

        # Translate the mask into indices into the data source
        inds = transpose(indices((nRows, nCols)), (1, 2, 0))
        inds = reshape(inds, (nRows * nCols, 2))
        inds = compress(not_equal(sMask.flat, 0), inds, 0).astype('s')

        self._size = inds.shape[0]

        # Now read the appropriate data into an array to calc
        # the statistics.
        #
        # TO DO:
        # Note that reading the entire training set at once
        # could be bad if it is defined for a large region. This
        # needs to be reworked using some type of iterator to avoid
        # having the entire training set in memory at once.

        data = zeros((inds.shape[0], nBands), self.image.typecode())
        for i in range(inds.shape[0]):
            d = self.image[inds[i][0], inds[i][1]].astype(data.typecode())
            data[i] = d
        (self.mean, self.cov) = mean_cov(data.astype(Float))
        self.invCov = inverse(self.cov)
        
        self.logDetCov = logDeterminant(self.cov)

        self._statsValid = 1

    def transformStatistics(self, m):
        '''
        Perform a linear transformation, m, on the statistics of the
        training set.

        USAGE: set.transform(m)
        '''

        from LinearAlgebra import *

        self.mean = dot(m, self.mean[:, NewAxis])[:, 0]
        self.cov = dot(m, dot(self.cov, transpose(m)))
        self.invCov = inverse(self.cov)
        
        try:
            self.logDetCov = log(determinant(self.cov))
        except OverflowError:
            self.logDetCov = sum(log(eigenvalues(self.cov)))

    def dump(self, fp):
        '''
        Dumps the TrainingSet object to a file stream.  Note that the
        image reference is replaced by the images file name.  It the
        responsibility of the loader to verify that the file name
        is replaced with an actual image object.
        '''
        from Numeric import *
        import pickle

        pickle.dump(self.image.fileName, fp)
        pickle.dump(self.index, fp)
        pickle.dump(self._size, fp)
        pickle.dump(self.classProb, fp)
        DumpArray(self.mask, fp)
        DumpArray(self.mean, fp)
        DumpArray(self.cov, fp)
        DumpArray(self.invCov, fp)
        pickle.dump(self.logDetCov, fp)
        
    def load(self, fp):
        '''
        Loads the TrainingSet object from a file stream.  The image
        member was probably replaced by the name of the image's source
        file before serialization.  The member should be replaced by
        the caller with an actual image object.
        '''
        from Numeric import *
        import pickle

        self.image = pickle.load(fp)
        self.index = pickle.load(fp)
        self._size = pickle.load(fp)
        self.classProb = pickle.load(fp)
        self.mask = LoadArray(fp)
        self.mean = LoadArray(fp)
        self.cov = LoadArray(fp)
        self.invCov = LoadArray(fp)
        self.logDetCov = pickle.load(fp)

def createTrainingSets(image, classMask, calcStats = 0):
    '''
    Create a list of TrainingSet objects from an indexed array.

    USAGE:  sets = createTrainingSets(classMask)

    ARGUMENTS:
        image               The image (MxNxB array or SpyFile) for
                            which the training sets are being defined.
        classMask           A rank-2 array whose elements are indices
                            of various spectral classes.
        calcStats           An optional parameter which, if non-zero,
                            causes statistics to be calculated for the
                            list of training sets.
    RETURN VALUE:
        sets                A list of TrainingSet objects

    The dimensions of classMask should be the same as the first two
    dimensions of the corresponding image. Values of zero in classMask
    are considered unlabeled and are not added to a training set.
    '''

    sets = []
    for i in range(1, max(classMask.flat)):
        if sum(equal(classMask, i).flat) > 0:
            ts = TrainingSet(image, classMask, i)
            if calcStats:
                ts.calcStatistics()
            sets.append(ts)
    return sets


def classifySpectrum(x, classes):
    '''
    Classify pixel into one of classes.

    USAGE: classIndex = classifySpectrum(x, classes)

    ARGUMENTS:
        x           The spectrum to classify
        classes     A list of TrainingSet objects with stats
    RETURN VALUE
        classIndex  The 'index' property of the most likely class
                    in classes.
    '''

    from math import log

    maxProb = -100000000000.
    maxClass = -1

    for i in range(len(classes)):
	cl = classes[i]
	delta = (x - cl.mean)[:, NewAxis]
	prob = log(cl.classProb) - 0.5 * cl.logDetCov		\
		- 0.5 * matrixmultiply(transpose(delta),	\
		matrixmultiply(cl.invCov, delta))
	if i == 0:
	    maxProb = prob[0,0]
	    maxClass = classes[0].index
	elif (prob[0,0] > maxProb):
	    maxProb = prob[0,0]
	    maxClass = classes[i].index
    return maxClass


def classifyImage(im, classes):
    '''
    Classify each image pixel into one of the specified classes.

    USAGE: classMap = classifyImage(data, classes)

    ARGUMENTS:
        data        An MxNxB Numeric array or a SpyFile object
        classes     A list of TrainingSet objects
    RETURN VALUE:
        classMap    A 2D array with dimensions same as data, whose
                    elements are the index property of the associated
                    TrainingSet object.
    '''

    from LinearAlgebra import *

    (nRows, nCols) = im.shape[:2]
    classMap = zeros((nRows, nCols), Int0)
    
    print 'Classifying image:'
    for i in range(nRows):
        print '\tClassifying: %5.1f%%' % (float(i) / nRows * 100.)
        for j in range(nCols):
            classMap[i, j] = classifySpectrum(im[i, j], classes)
    print '\tDone.'

    return classMap



def ndvi(data, red, nir):
    '''
    Calculate the Normalized Difference Vegetation Index (NDVI) for the
    given data.

    USAGE: vi = ndvi(data, red, nir)

    ARGUMENTS:
        data        The array or SpyFile for which to calc. the index
        red         An integer or range integers specifying the red bands.
        nir         An integer or range integers specifying the near
                        infrared bands.
    RETURN VALUE:
        An array containing NDVI values for each corresponding element
        of data in the range [0.0, 1.0].
    '''

    r = data[:, :, red].astype(Float)
    if len(r.shape) == 3 and r.shape[2] > 1:
        r = sum(r, 2) / r.shape[2]
    n = data[:, :, nir].astype(Float)
    if len(n.shape) == 3 and n.shape[2] > 1:
        n = sum(n, 2) / n.shape[2]

    return (n - r) / (n + r)


def bhattacharyyaDistance(a, b):
    '''
    Calulate the Bhattacharyya distance between two classes.

    USAGE:  bd = bhattacharyyaDistance(a, b)

    ARGUMENTS:
        (a, b)              The classes for which to determine the
                            B-distance.
    RETURN VALUE:
        bd                  The B-distance between a and b.
    '''
    
    terms = bDistanceTerms(a, b)
    return terms[0] + terms[1]


bDistance = bhattacharyyaDistance

def bDistanceTerms(a, b):
    '''
    Calulate the linear and quadratic terms of the Bhattacharyya distance
    between two classes.

    USAGE:  (linTerm, quadTerm = bDistanceTerms(a, b)

    ARGUMENTS:
        (a, b)              The classes for which to determine the
                            B-distance.
    RETURN VALUE:
                            A 2-tuple of the linear and quadratic terms
    '''
    from math import exp

    m = a.mean - b.mean
    avgCov = (a.cov + b.cov) / 2

    linTerm = (1/8.) * matrixmultiply(transpose(m), \
        matrixmultiply(inverse(avgCov), m))

    quadTerm = 0.5 * (logDeterminant(avgCov) \
                      - 0.5 * a.logDetCov \
                      - 0.5 * b.logDetCov)

    return (linTerm, float(quadTerm))


def transformImage(image, matrix):
    '''
    Perform linear transformation on all pixels in an image.

    USAGE: trData = transformImage(im, matrix)

    ARGUMENTS:
        im              Image data to transform
        matrix          The linear transform to apply
    RETURN VALUE:
        trData          The transformed image data

    Reads in an entire image, transforming each pixel by matrix on input.
    This function can also be used to transform each pixel in a rank-3
    NumPy array.
    '''
    
    (M, N, B) = image.shape
    xImage = zeros((M, N, matrix.shape[0]), matrix.typecode())
    
    for i in range(M):
    	for j in range(N):
	    xImage[i, j] = matrixmultiply(matrix, image[i, j]  \
	    				.astype(matrix.typecode()))
    return xImage



def orthogonalize(vecs, start = 0):
    '''
    Perform Gram-Schmidt Orthogonalization on a set of vectors.

    USAGE:  basis = gso(vecs [, start = 0])

    RETURN VALUE:
        basis           An orthonormal basis spanning vecs.
        
    If start is specified, it is assumed that vecs[:start] are already
    orthonormal.
    '''

    from LinearAlgebra import *
    from math import sqrt
    
    (M, N) = vecs.shape
    basis = array(transpose(vecs))
    eye = identity(N).astype(Float)
    if start == 0:
	basis[:, 0] /= sqrt(dot(basis[:, 0], basis[:, 0]))
	start = 1
    
    for i in range(start, M):
	v = basis[:, i] / sqrt(dot(basis[:, i], basis[:, i]))
    	U = basis[:, :i]
	P = eye - dot(U, dot(inverse(dot(transpose(U), U)), transpose(U)))
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

    from LinearAlgebra import inverse

    assert members.shape[1] == data.shape[2], \
           'Matrix dimensions are not aligned.'

    # Calculate the pseudo inverse
    pi = dot(members, transpose(members))
    pi = dot(inverse(pi), members)

    (M, N, B) = data.shape
    unmixed = zeros((M, N, members.shape[0]), Float)
    for i in range(M):
        for j in range(N):
            unmixed[i, j] = dot(pi, data[i,j])
    return unmixed


def spectralAngles(data, members):
    '''
    Perform spectral angle mapping of data.

    USAGE: angles = spectralAngles(data, members)

    ARGUMENTS:
        data            MxNxB image data
        members         CxB array of spectral endmembers
    RETURN VALUE:
        angles          An MxNxC array of spectral angles.

    
    Calculates the spectral angles between each vector in data and each
    of the endmembers.  The output of this function (angles) can be used
    to classify the data by minimum spectral angle by calling argmin(angles).
    This function currently does not use second order statistics.
    '''

    assert members.shape[1] == data.shape[2], \
           'Matrix dimensions are not aligned.'    

    (M, N, B) = data.shape
    m = array(members)
    C = m.shape[0]

    # Normalize endmembers
    for i in range(C):
        m[i] /= sqrt(dot(m[i], m[i]))
    
    angles = zeros((M, N, C), Float)
    
    for i in range(M):
        for j in range(N):
            v = data[i, j].astype(Float)
            v = v / sqrt(dot(v, v))
            for k in range(C):
                angles[i, j, k] = dot(v, m[k])

    return arccos(angles)
            
