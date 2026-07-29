'''
Tests of linear transforms of spectral data & data files.

The unit tests in this module open the sample image "92AV3C.lan" (bundled
under `tests/data/`) and verify that LinearTransform objects created with
SpyFile and numpy.ndarray objects yield the correct values for known image
data values.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_transforms.py
'''

from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from spectral.algorithms.transforms import LinearTransform


@pytest.fixture(params=['spyfile', 'ndarray', 'scaled'])
def transform_scenario(request, av3c_image):
    '''Returns (image, datum, value) for each of the scenarios under test.

    A fresh `av3c_image` is opened for every scenario (since `av3c_image` is
    function-scoped) so that setting `scale_factor` on the SpyFile in the
    "scaled" scenario doesn't affect the other scenarios.
    '''
    datum = (99, 99, 99)
    value = 2057.0
    if request.param == 'spyfile':
        return (av3c_image, datum, value)
    elif request.param == 'ndarray':
        return (av3c_image.load(), datum, value)
    else:
        av3c_image.scale_factor = 10000.0
        return (av3c_image, datum, value / 10000.0)


@pytest.fixture
def transform_setup(transform_scenario):
    (image, datum, value) = transform_scenario
    scalar = 10.
    matrix = scalar * np.identity(image.shape[2], dtype='f8')
    pre = 37.
    post = 51.
    return SimpleNamespace(image=image, datum=datum, value=value,
                           scalar=scalar, matrix=matrix, pre=pre, post=post)


class TestLinearTransform:
    '''Tests that LinearTransform objects produce correct values.'''

    def test_scalar_multiply(self, transform_setup):
        s = transform_setup
        (i, j, k) = s.datum
        transform = LinearTransform(s.scalar)
        result = transform(s.image[i, j])[k]
        assert_almost_equal(result, s.scalar * s.value)

    def test_pre_scalar_multiply(self, transform_setup):
        s = transform_setup
        (i, j, k) = s.datum
        transform = LinearTransform(s.scalar, pre=s.pre)
        result = transform(s.image[i, j])[k]
        assert_almost_equal(result, s.scalar * (s.pre + s.value))

    def test_scalar_multiply_post(self, transform_setup):
        s = transform_setup
        (i, j, k) = s.datum
        transform = LinearTransform(s.scalar, post=s.post)
        result = transform(s.image[i, j])[k]
        assert_almost_equal(result, s.scalar * s.value + s.post)

    def test_pre_scalar_multiply_post(self, transform_setup):
        s = transform_setup
        (i, j, k) = s.datum
        transform = LinearTransform(s.scalar, pre=s.pre, post=s.post)
        result = transform(s.image[i, j])[k]
        assert_almost_equal(result,
                            s.scalar * (s.pre + s.value) + s.post)

    def test_matrix_multiply(self, transform_setup):
        s = transform_setup
        (i, j, k) = s.datum
        transform = LinearTransform(s.matrix)
        result = transform(s.image[i, j])[k]
        assert_almost_equal(result, s.scalar * s.value)

    def test_pre_matrix_multiply(self, transform_setup):
        s = transform_setup
        (i, j, k) = s.datum
        transform = LinearTransform(s.matrix, pre=s.pre)
        result = transform(s.image[i, j])[k]
        assert_almost_equal(result, s.scalar * (s.pre + s.value))

    def test_matrix_multiply_post(self, transform_setup):
        s = transform_setup
        (i, j, k) = s.datum
        transform = LinearTransform(s.matrix, post=s.post)
        result = transform(s.image[i, j])[k]
        assert_almost_equal(result, s.scalar * s.value + s.post)

    def test_pre_matrix_multiply_post(self, transform_setup):
        s = transform_setup
        (i, j, k) = s.datum
        transform = LinearTransform(s.matrix, pre=s.pre, post=s.post)
        result = transform(s.image[i, j])[k]
        assert_almost_equal(result,
                            s.scalar * (s.pre + s.value) + s.post)
