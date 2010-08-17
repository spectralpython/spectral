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

class Iterator:
    '''
    Base class for iterators over pixels (spectra).
    '''
    def __init__(self):
        pass
    def __iter__(self):
        raise NotImplementedError('Must override __iter__ in child class.')
    def getNumElements(self):
        raise NotImplementedError('Must override getNumElements in child class.')
    def getNumBands(self):
        raise NotImplementedError('Must override getNumBands in child class.')

class ImageIterator(Iterator):
    '''
    An iterator over all pixels in an image.
    '''
    def __init__(self, im):
        self.image = im
        self.numElements = im.shape[0] * im.shape[1]
    def getNumElements(self):
        return self.numElements
    def getNumBands(self):
        return self.image.shape[2]
    def __iter__(self):
        from spectral import status
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
    def __init__(self, im, mask, index = None):
        self.image = im
        self.index = index
        # Get the proper mask for the training set
        if index:
            self.mask = numpy.equal(mask, index)
        else:
            self.mask = not_equal(mask, 0)
        self.numElements = sum(self.mask.ravel())
    def getNumElements(self):
        return self.numElements
    def getNumBands(self):
        return self.image.shape[2]
    def __iter__(self):
        from spectral import status
	from spectral.io import typecode
        from numpy import transpose, indices, reshape, compress, not_equal
        typechar = typecode(self.image)
        (nRows, nCols, nBands) = self.image.shape

        # Translate the mask into indices into the data source
        inds = transpose(indices((nRows, nCols)), (1, 2, 0))
        inds = reshape(inds, (nRows * nCols, 2))
        inds = compress(not_equal(self.mask.ravel(), 0), inds, 0).astype('h')

        for i in range(inds.shape[0]):
            sample = self.image[inds[i][0], inds[i][1]].astype(typechar)
            if len(sample.shape) == 3:
                sample.shape = (sample.shape[2],)
            (self.row, self.col) = inds[i][:2]
            yield sample

def iterator(image, mask = None, index = None):
    '''
    Function returning an iterator over pixels in the image.
    '''

    if isinstance(image, Iterator):
        return image
    elif mask != None:
        return ImageMaskIterator(image, mask, index)
    else:
        return ImageIterator(image)


def mean_cov(image, mask = None, index = None):
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
    can be an Iterator, a SpyFile object, or an MxNxB array.
    '''
    import spectral
    from spectral import status
    from numpy import zeros, transpose, dot
    from numpy.oldnumeric import NewAxis
    
    if not isinstance(image, Iterator):
        it = iterator(image, mask, index)
    else:
        it = image

    nSamples = it.getNumElements()
    B = it.getNumBands()
    
    sumX = zeros((B,), float)
    sumX2 = zeros((B, B), float)
    count = 0
    
    statusInterval = max(1, nSamples / 100)
    status.displayPercentage('Covariance.....')
    for x in it:
        if not count % statusInterval:
            status.updatePercentage(float(count) / nSamples * 100.)
        count += 1
        sumX += x
        x = x[:, NewAxis].astype(float)
        sumX2 += dot(x, transpose(x))
    mean = sumX / count
    sumX = sumX[:, NewAxis]
    cov = (sumX2 - dot(sumX, transpose(sumX)) / float(count)) / float(count - 1)
    status.endPercentage()
    return (mean, cov, count)

def covariance(*args):
    return mean_cov(*args)[1]

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
    
    from numpy import sqrt, sum
    
    (M, N, B) = image.shape
    
    (mean, cov, count) = mean_cov(image)
    (L, V) = numpy.linalg.eig(cov)

    #  Normalize eigenvectors
    V = V / sqrt(sum(V * V, 0))

    # numpy stores eigenvectors in columns
    V = V.transpose()

    return (L, V, mean, cov)


def linearDiscriminant(classes):
    '''
    Solve Fisher's linear discriminant for eigenvalues and eigenvectors.

    USAGE: (L, V, CB, CW) = linearDiscriminant(classes)

    Determines the solution to the generalized eigenvalue problem
    
            cov_b * x = lambda * cov_w * x
            
    Since cov_w is normally invertable, the reduces to
    
            (inv(cov_w) * cov_b) * x = lambda * x
            
    The return value is a 4-tuple containing the vector of eigenvalues,
    a matrix of the corresponding eigenvectors, the between-class
    covariance matrix, and the within-class covariance matrix.
    '''

    from numpy import zeros, dot, transpose, diagonal
    from numpy.linalg import inv, eig
    from numpy.oldnumeric import NewAxis
    import math

    C = len(classes)		# Number of training sets
    rank = len(classes) - 1

    # Calculate total # of training pixels and total mean
    N = 0
    B = None            # Don't know number of bands yet
    mean = None
    for s in classes:
        if mean == None:
            B = s.numBands
            mean = zeros(B, float)
	N += s.size()
        if not hasattr(s, 'stats'):
            s.calcStatistics()
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
    d = diagonal(v)
    for i in range(vecs.shape[0]):
    	vecs[i, :] /= math.sqrt(d[i].real)
    	
    return (vals.real, vecs.real, cov_b, cov_w)

# Alias for Linear Discriminant Analysis (LDA)
lda = linearDiscriminant


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

    import numpy.oldnumeric as Numeric

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
    from numpy.oldnumeric.linear_algebra import eigenvalues
    return sum(numpy.log(eigenvalues(x)))

class GaussianStats:
    def __init__(self):
        self.numSamples = 0

class TrainingClass:
    def __init__(self, image, mask, index = 0, classProb = 1.0):
        '''
        Define a Training Class by selecting a set of vectors from image
        by applying the given mask.

        USAGE: ts = TrainingClass(image, mask [, index] [classProb = 1.0)

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
        self.numBands = image.shape[2]
        self.mask = mask
        self.index = index
        self.classProb = classProb

        self._statsValid = 0
        self._size = 0

    def __iter__(self):
        it = ImageMaskIterator(self.image, self.mask, self.index)
        for i in it:
            yield i

    def statsValid(self, tf):
        '''
        Set statistics for the TrainingClass to be valid or invalid.

        USAGE: tset.statsValid(bool)

        This function is intended to be called with a zero argument
        when the set's statistics become invalid (e.g., the data
        source or mask changes).
        '''
        self._statsValid = tf

    def size(self):
        '''Return the number of pixels in the training set.'''
        from numpy import sum, equal

        # If the stats are invalid, the number of pixels in the
        # training set may have changed.
        if self._statsValid:
            return self._size

        if self.index:
            return sum(equal(self.mask, self.index).ravel())
        else:
            return sum(not_equal(self.mask, 0).ravel())        

    def calcStatistics(self):
        '''
        Calculates statistic for the class.
        '''
        import math
        from numpy.linalg import inv, det

        self.stats = GaussianStats()
        (self.stats.mean, self.stats.cov, self.stats.numSamples) = \
                          mean_cov(self.image, self.mask, self.index)
        self.stats.invCov = inv(self.stats.cov)
        self.stats.logDetCov = logDeterminant(self.stats.cov)
        self._size = self.stats.numSamples
        self._statsValid = 1

    def transform(self, m):
        '''
        Perform a linear transformation, m, on the statistics of the
        training set.

        USAGE: set.transform(m)
        '''

        from numpy import dot, transpose
        from numpy.linalg import det, inv
        from numpy.oldnumeric import NewAxis
        import math
        from spectral.io.spyfile import TransformedImage

        self.stats.mean = dot(m, self.stats.mean[:, NewAxis])[:, 0]
        self.stats.cov = dot(m, dot(self.stats.cov, transpose(m)))
        self.stats.invCov = inv(self.stats.cov)
        
        try:
            self.stats.logDetCov = math.log(det(self.stats.cov))
        except OverflowError:
            self.stats.logDetCov = logDeterminant(self.stats.cov)

        self.numBands = m.shape[0]
        self.image = TransformedImage(m, self.image)

    def dump(self, fp):
        '''
        Dumps the TrainingClass object to a file stream.  Note that the
        image reference is replaced by the images file name.  It the
        responsibility of the loader to verify that the file name
        is replaced with an actual image object.
        '''
        import pickle

        pickle.dump(self.image.fileName, fp)
        pickle.dump(self.index, fp)
        pickle.dump(self._size, fp)
        pickle.dump(self.classProb, fp)
        DumpArray(self.mask, fp)
        DumpArray(self.stats.mean, fp)
        DumpArray(self.stats.cov, fp)
        DumpArray(self.stats.invCov, fp)
        pickle.dump(self.stats.logDetCov, fp)
        
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
        self.classProb = pickle.load(fp)
        self.mask = LoadArray(fp)
        self.stats.mean = LoadArray(fp)
        self.stats.cov = LoadArray(fp)
        self.stats.invCov = LoadArray(fp)
        self.stats.logDetCov = pickle.load(fp)
        self.stats.numSamples = self._size

class SampleIterator:
    '''An iterator over all samples of all classes in a TrainingData object.'''
    def __init__(self, trainingData):
        self.classes = trainingData
    def __iter__(self):
        for cl in self.classes:
            for sample in cl:
                yield sample
            
class TrainingClassSet:
    def __init__(self):
        self.classes = {}
        self.numBands = None
    def __getitem__(self, i):
        '''Returns the class having index i.'''
        return self.classes[i]
    def __len__(self):
        return len(self.classes)
    def addClass(self, cl):
        if self.classes.has_key(cl.index):
            raise 'Attempting to add class with duplicate index.'
        self.classes[cl.index] = cl
        if not self.numBands:
            self.numBands = cl.numBands
    def transform(self, M):
        '''Apply linear transform, M, to all training classes.'''
        for cl in self.classes.values():
            cl.transform(M)
        self.numBands = M.shape[0]
        
    def __iter__(self):
        '''
        Returns an iterator over all TrainingClass objects.
        '''
        for cl in self.classes.values():
            yield cl
    def allSamples(self):
        return SampleIterator(self)
        
def createTrainingClasses(image, classMask, calcStats = 0, indices = None):
    '''
    Create a list of TrainingClass objects from an indexed array.

    USAGE:  sets = createTrainingClasses(classMask)

    ARGUMENTS:
        image               The image (MxNxB array or SpyFile) for
                            which the training sets are being defined.
        classMask           A rank-2 array whose elements are indices
                            of various spectral classes.
        calcStats           An optional parameter which, if non-zero,
                            causes statistics to be calculated for the
                            list of training sets.
    RETURN VALUE:
        sets                A list of TrainingClass objects

    The dimensions of classMask should be the same as the first two
    dimensions of the corresponding image. Values of zero in classMask
    are considered unlabeled and are not added to a training set.
    '''

    classIndices = set(classMask.ravel())
    classes = TrainingClassSet()
    for i in classIndices:
        if i == 0:
            # Index 0 denotes unlabled pixel
            continue
        elif indices and not i in indices:
            continue
        cl = TrainingClass(image, classMask, i)
        if calcStats:
            cl.calcStatistics()
        classes.addClass(cl)
    return classes


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

    r = data[:, :, red].astype(float)
    if len(r.shape) == 3 and r.shape[2] > 1:
        r = sum(r, 2) / r.shape[2]
    n = data[:, :, nir].astype(float)
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

    linTerm = (1/8.) * dot(transpose(m), \
        dot(inv(avgCov), m))

    quadTerm = 0.5 * (logDeterminant(avgCov) \
                      - 0.5 * a.stats.logDetCov \
                      - 0.5 * b.stats.logDetCov)

    return (linTerm, float(quadTerm))


def transformImage(matrix, image):
    '''
    Perform linear transformation on all pixels in an image.

    USAGE: trData = transformImage(im, matrix)

    ARGUMENTS:
        matrix          The linear transform to apply
        im              Image data to transform
    RETURN VALUE:
        trData          The transformed image

    If the image argument is a SpyFile object, a TransformedImage object
    is returned.  If image is a Numeric array, an array with all pixels
    transformed is returned.
    '''
    from spectral.io.spyfile import TransformedImage
    from numpy.oldnumeric import ArrayType
    from spectral.io.spyfile import SpyFile

    if isinstance(image, SpyFile):
        return TransformedImage(matrix, image)
    elif isinstance(image, ArrayType):
        (M, N, B) = image.shape
        xImage = numpy.zeros((M, N, matrix.shape[0]), float)
        
        for i in range(M):
            for j in range(N):
                xImage[i, j] = numpy.dot(matrix, image[i, j].astype(float))
        return xImage
    else:
        raise 'Unrecognized image type passed to transformImage.'

def orthogonalize(vecs, start = 0):
    '''
    Perform Gram-Schmidt Orthogonalization on a set of vectors.

    USAGE:  basis = orthogonalize(vecs [, start = 0])

    RETURN VALUE:
        basis           An orthonormal basis spanning vecs.
        
    If start is specified, it is assumed that vecs[:start] are already
    orthonormal.
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
    from numpy.linag import inv

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
            unmixed[i, j] = dot(pi, data[i,j].astype(float))
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
    from numpy import dot, zeros, arccos

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
            
