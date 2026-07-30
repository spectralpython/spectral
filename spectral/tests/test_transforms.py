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
from numpy.testing import assert_allclose, assert_almost_equal
import pytest

from spectral.algorithms.transforms import LinearTransform

# Deterministic (non-square, non-commuting) matrices/offsets used by
# TestChain to verify that `f1.chain(f2)` reproduces `f2(f1(X))`.
MATRIX_1 = np.array([[1., 2., 0.], [0., 1., 3.], [2., 0., 1.], [1., 1., 1.]])
PRE_1 = np.array([1., 0., -1.])
POST_1 = np.array([0.5, 1.5, -0.5, 2.0])

MATRIX_2 = np.array([[1., 0., -1., 2.], [0., 2., 1., 0.]])
PRE_2 = np.array([2., -1., 0., 1.])
POST_2 = np.array([3., -2.])

CHAIN_X = np.array([[1., 2., 3.], [0., -1., 2.], [4., 0., 1.]])


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

    def test_call_raises_typeerror_for_unsupported_input(self):
        transform = LinearTransform(2.0)
        with pytest.raises(TypeError):
            transform(object())


class TestChain:
    '''Tests `LinearTransform.chain()`.

    `f1.chain(f2)` composes two transforms so that applying the result to
    `X` is equivalent to applying `f1` then `f2`: `f3(X) == f2(f1(X))`. That
    equivalence is the contract each test below verifies, rather than
    depending on `chain()`'s internal formula.
    '''

    def test_full_pre_post_non_square(self):
        f1 = LinearTransform(MATRIX_1, pre=PRE_1, post=POST_1)
        f2 = LinearTransform(MATRIX_2, pre=PRE_2, post=POST_2)
        f3 = f1.chain(f2)
        assert_allclose(f3(CHAIN_X), f2(f1(CHAIN_X)))

    def test_no_pre_post(self):
        f1 = LinearTransform(MATRIX_1)
        f2 = LinearTransform(MATRIX_2)
        f3 = f1.chain(f2)
        assert_allclose(f3(CHAIN_X), f2(f1(CHAIN_X)))

    def test_only_first_transform_has_pre(self):
        f1 = LinearTransform(MATRIX_1, pre=PRE_1)
        f2 = LinearTransform(MATRIX_2)
        f3 = f1.chain(f2)
        assert_allclose(f3(CHAIN_X), f2(f1(CHAIN_X)))

    def test_only_first_transform_has_post(self):
        f1 = LinearTransform(MATRIX_1, post=POST_1)
        f2 = LinearTransform(MATRIX_2)
        f3 = f1.chain(f2)
        assert_allclose(f3(CHAIN_X), f2(f1(CHAIN_X)))

    def test_only_second_transform_has_pre(self):
        f1 = LinearTransform(MATRIX_1)
        f2 = LinearTransform(MATRIX_2, pre=PRE_2)
        f3 = f1.chain(f2)
        assert_allclose(f3(CHAIN_X), f2(f1(CHAIN_X)))

    def test_only_second_transform_has_post(self):
        f1 = LinearTransform(MATRIX_1)
        f2 = LinearTransform(MATRIX_2, post=POST_2)
        f3 = f1.chain(f2)
        assert_allclose(f3(CHAIN_X), f2(f1(CHAIN_X)))

    def test_square_non_commuting_matrices(self):
        '''Since matrix multiplication doesn't commute, this would catch
        `chain()` combining the two transforms' matrices in the wrong
        order even though both have the same input/output dimension.'''
        m1 = np.array([[1., 2.], [0., 1.]])
        m2 = np.array([[0., 1.], [1., 0.]])
        f1 = LinearTransform(m1, pre=[1., -1.], post=[2., 0.])
        f2 = LinearTransform(m2, pre=[0., 1.], post=[-1., 3.])
        x = np.array([[1., 2.], [3., -1.]])
        f3 = f1.chain(f2)
        assert_allclose(f3(x), f2(f1(x)))

    def test_scalar_then_matrix(self):
        f1 = LinearTransform(2.0)
        f2 = LinearTransform(MATRIX_2)
        x = np.array([[1., 2., 3., 4.], [0., -1., 2., 1.]])
        f3 = f1.chain(f2)
        assert_allclose(f3(x), f2(f1(x)))

    def test_matrix_then_scalar(self):
        f1 = LinearTransform(MATRIX_1)
        f2 = LinearTransform(3.0)
        f3 = f1.chain(f2)
        assert_allclose(f3(CHAIN_X), f2(f1(CHAIN_X)))

    def test_chain_accepts_ndarray_as_transform(self):
        '''Passing a plain ndarray (instead of a LinearTransform) as the
        `transform` argument should be wrapped automatically.'''
        f1 = LinearTransform(MATRIX_1, pre=PRE_1, post=POST_1)
        f2 = LinearTransform(MATRIX_2)
        f3 = f1.chain(MATRIX_2)
        assert_allclose(f3(CHAIN_X), f2(f1(CHAIN_X)))

    def test_mismatched_dimensions_raises(self):
        f1 = LinearTransform(MATRIX_1)          # dim_out = 4
        f2 = LinearTransform(np.zeros((2, 5)))  # dim_in = 5
        with pytest.raises(Exception):
            f1.chain(f2)
