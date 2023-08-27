'''
Runs unit tests for classification routines.

To run the unit tests, type the following from the system command line:

    # python -m spectral.tests.classifiers
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import spectral as spy
from numpy.testing import assert_allclose
from .spytest import SpyTest
from spectral.tests import testdir


class ClassifierTest(SpyTest):
    '''Tests various classfication functions.'''

    def setup(self):
        if not os.path.isdir(testdir):
            os.mkdir(testdir)
        self.image = spy.open_image('92AV3C.lan')
        self.data = self.image.load()
        self.gt = spy.open_image('92AV3GT.GIS').read_band(0)
        self.ts = spy.create_training_classes(self.data, self.gt,
                                              calc_stats=True)
        self.class_filename = os.path.join(testdir, '92AV3C.classes')

    def test_save_training_sets(self):
        '''Test that TrainingClassSet data can be saved without exception.'''
        ts = spy.create_training_classes(self.data, self.gt, calc_stats=True)
        ts.save(self.class_filename)

    def test_load_training_sets(self):
        '''Test that the data loaded is the same as was saved.'''
        ts = spy.create_training_classes(self.data, self.gt, calc_stats=True)
        ts.save(self.class_filename)
        ts2 = spy.load_training_sets(self.class_filename, image=self.data)
        ids = list(ts.classes.keys())
        for id in ids:
            s1 = ts[id]
            s2 = ts2[id]
            assert (s1.index == s2.index)
            np.testing.assert_almost_equal(s1.class_prob, s2.class_prob)
            assert_allclose(s1.stats.mean, s2.stats.mean)
            assert_allclose(s1.stats.cov, s2.stats.cov)
            np.testing.assert_equal(s1.stats.nsamples, s2.stats.nsamples)

    def test_gmlc_spectrum_image_equal(self):
        '''Tests that classification of spectrum is same as from image.'''
        gmlc = spy.GaussianClassifier(self.ts, min_samples=600)
        data = self.data[20: 30, 30: 40, :]
        assert (gmlc.classify_spectrum(data[2, 2]) ==
                gmlc.classify_image(data)[2, 2])

    def test_gmlc_classify_spyfile_runs(self):
        '''Tests that GaussianClassifier classifies a SpyFile object.'''
        gmlc = spy.GaussianClassifier(self.ts, min_samples=600)
        gmlc.classify_image(self.image)

    def test_gmlc_classify_transformedimage_runs(self):
        '''Tests that GaussianClassifier classifies a TransformedImage object.'''
        pc = spy.principal_components(self.data).reduce(num=3)
        ximg = pc.transform(self.image)
        ts = spy.create_training_classes(pc.transform(self.data), self.gt,
                                         calc_stats=True)
        gmlc = spy.GaussianClassifier(ts)
        gmlc.classify_image(ximg)

    def test_gmlc_classify_ndarray_transformedimage_equal(self):
        '''Gaussian classification of an ndarray and TransformedImage are equal'''
        pc = spy.principal_components(self.data).reduce(num=3)
        ximg = pc.transform(self.image)
        ts = spy.create_training_classes(pc.transform(self.data), self.gt,
                                         calc_stats=True)
        gmlc = spy.GaussianClassifier(ts)
        cl_ximg = gmlc.classify_image(ximg)
        cl_ndarray = gmlc.classify_image(pc.transform(self.data))
        assert (np.all(cl_ximg == cl_ndarray))

    def test_mahalanobis_class_mean(self):
        '''Test that a class's mean spectrum is classified as that class.
        Note this assumes that class priors are equal.
        '''
        mdc = spy.MahalanobisDistanceClassifier(self.ts)
        cl = mdc.classes[0]
        assert (mdc.classify(cl.stats.mean) == cl.index)

    def test_mahalanobis_classify_spyfile_runs(self):
        '''Mahalanobis classifier works with a SpyFile object.'''
        mdc = spy.MahalanobisDistanceClassifier(self.ts)
        mdc.classify_image(self.image)

    def test_mahalanobis_classify_transformedimage_runs(self):
        '''Mahalanobis classifier works with a TransformedImage object.'''
        pc = spy.principal_components(self.data).reduce(num=3)
        ximg = pc.transform(self.image)
        ts = spy.create_training_classes(pc.transform(self.data), self.gt,
                                         calc_stats=True)
        gmlc = spy.MahalanobisDistanceClassifier(ts)
        gmlc.classify_image(ximg)

    def test_mahalanobis_classify_ndarray_transformedimage_equal(self):
        '''Mahalanobis classification of ndarray and TransformedImage are equal'''
        pc = spy.principal_components(self.data).reduce(num=3)
        ximg = pc.transform(self.image)
        ts = spy.create_training_classes(pc.transform(self.data), self.gt,
                                         calc_stats=True)
        mdc = spy.GaussianClassifier(ts)
        cl_ximg = mdc.classify_image(ximg)
        cl_ndarray = mdc.classify_image(pc.transform(self.data))
        assert (np.all(cl_ximg == cl_ndarray))

    def test_perceptron_learns_and(self):
        '''Test that 2x1 network can learn the logical AND function.'''
        from spectral.algorithms.perceptron import test_and
        (success, p) = test_and(stdout=None)
        assert (success)
        
    def test_perceptron_learns_xor(self):
        '''Test that 2x2x1 network can learn the logical XOR function.'''
        from spectral.algorithms.perceptron import test_xor231
        # XOR isn't guaranteed to converge so try at lease a few times
        for i in range(10):
            (success, p) = test_xor231(3000, stdout=None)
            if success is True:
                return
        assert (False)

    def test_perceptron_learns_xor_222(self):
        '''Test that 2x2x2 network can learn the logical XOR function.'''
        from spectral.algorithms.perceptron import test_xor222
        # XOR isn't guaranteed to converge so try at lease a few times
        for i in range(10):
            (success, p) = test_xor222(3000, stdout=None)
            if success is True:
                return
        assert (False)

    def test_perceptron_learns_image_classes(self):
        '''Test that perceptron can learn image class means.'''
        fld = spy.linear_discriminant(self.ts)
        xdata = fld.transform(self.data)
        classes = spy.create_training_classes(xdata, self.gt)
        nfeatures = xdata.shape[-1]
        nclasses = len(classes)
        for i in range(10):
            p = spy.PerceptronClassifier([nfeatures, 20, 8, nclasses])
            success = p.train(classes, 1, 5000, batch=1, momentum=0.3,
                              rate=0.3)
            if success is True:
                return
        assert (False)

    def test_mahalanobis_spectrum_image_equal(self):
        '''Tests that classification of spectrum is same as from image.'''
        mdc = spy.MahalanobisDistanceClassifier(self.ts)
        data = self.data[20: 30, 30: 40, :]
        assert (mdc.classify_spectrum(data[2, 2]) ==
                mdc.classify_image(data)[2, 2])


def run():
    print('\n' + '-' * 72)
    print('Running classifier tests.')
    print('-' * 72)
    test = ClassifierTest()
    test.run()


if __name__ == '__main__':
    from spectral.tests.run import parse_args, reset_stats, print_summary
    parse_args()
    reset_stats()
    run()
    print_summary()
