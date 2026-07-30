'''Tests of k-means clustering.

`spy.kmeans` has two entirely separate implementations depending on the type
of its `image` argument: an ndarray input is delegated to `kmeans_ndarray`,
while any other (e.g., SpyFile-like) input is handled by a slower, pure
Python loop directly in `kmeans`. Both branches are exercised here.

All tests use a small, noise-free synthetic image made up of a handful of
well-separated blocks of identical pixel values, so that the correct
clustering is known in advance and convergence is deterministic.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_clustering.py
'''

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import spectral as spy
from spectral.algorithms.clustering import kmeans_ndarray
from spectral.utilities.errors import NaNValueError

# (row_slice, pixel_value) for each of the 3 blocks making up
# `three_cluster_image`. Since every pixel in a block has the identical
# value given here, cluster means should converge exactly to these values.
BLOCKS = [(slice(0, 2), 0.0), (slice(2, 4), 10.0), (slice(4, 6), 20.0)]


class NonNdarrayImage:
    '''Wraps an ndarray without subclassing it.

    `kmeans` dispatches on `isinstance(image, np.ndarray)` to decide which
    of its two implementations to use, so a plain ndarray can't be used to
    exercise the non-ndarray (SpyFile-like) branch. This class exposes just
    enough of the interface that branch relies on (`shape` and pixel
    indexing) while deliberately not being an ndarray instance.
    '''
    def __init__(self, data):
        self._data = data
        self.shape = data.shape

    def __getitem__(self, key):
        return self._data[key]


@pytest.fixture
def three_cluster_image():
    data = np.empty((6, 6, 2), dtype=float)
    for (rows, value) in BLOCKS:
        data[rows] = value
    return data


def assert_valid_clustering(class_map, centers, blocks=BLOCKS):
    '''Asserts that each block was assigned a single, distinct cluster label
    and that the corresponding cluster center matches the block's value.'''
    class_map = np.asarray(class_map)
    seen_labels = set()
    for (rows, value) in blocks:
        labels = set(class_map[rows].ravel().tolist())
        assert len(labels) == 1, \
            'Expected a single cluster label within block %r, got %r' % (
                rows, labels)
        label = labels.pop()
        assert label not in seen_labels, \
            'Two different blocks were assigned the same cluster label.'
        seen_labels.add(label)
        assert_allclose(centers[label], value)


class TestKmeansNdarray:
    '''Tests `kmeans_ndarray`, used when `kmeans` is passed an ndarray.'''

    def test_converges_to_expected_clusters(self, three_cluster_image):
        (class_map, centers) = kmeans_ndarray(three_cluster_image,
                                              nclusters=3, max_iterations=20)
        assert_valid_clustering(class_map, centers)

    @pytest.mark.parametrize('distance', ['L1', 'L2', spy.L1, spy.L2])
    def test_distance_options(self, three_cluster_image, distance):
        (class_map, centers) = kmeans_ndarray(three_cluster_image,
                                              nclusters=3, distance=distance)
        assert_valid_clustering(class_map, centers)

    def test_start_clusters(self, three_cluster_image):
        '''Cluster labels should follow the order of the given start
        centers rather than the (differently-ordered) default box-diagonal
        initialization, proving `start_clusters` was actually used.'''
        start = np.array([[20., 20.], [0., 0.], [10., 10.]])
        (class_map, centers) = kmeans_ndarray(three_cluster_image,
                                              nclusters=3,
                                              start_clusters=start)
        assert_valid_clustering(class_map, centers)
        assert class_map[0, 0] == 1
        assert class_map[2, 0] == 2
        assert class_map[4, 0] == 0

    def test_start_clusters_wrong_shape_raises(self, three_cluster_image):
        bad_start = np.zeros((2, three_cluster_image.shape[2]))
        with pytest.raises(AssertionError):
            kmeans_ndarray(three_cluster_image, nclusters=3,
                           start_clusters=bad_start)

    def test_compare_kwarg_terminates_iteration(self, three_cluster_image):
        calls = []

        def compare(old_clusters, clusters):
            calls.append((old_clusters.copy(), clusters.copy()))
            return True

        kmeans_ndarray(three_cluster_image, nclusters=3, max_iterations=20,
                       compare=compare)
        assert len(calls) == 1

    def test_frames_kwarg_records_each_iteration(self, three_cluster_image):
        frames = []
        (class_map, centers) = kmeans_ndarray(three_cluster_image,
                                              nclusters=3, max_iterations=20,
                                              frames=frames)
        assert len(frames) >= 1
        assert_array_equal(frames[-1], class_map)

    def test_frames_kwarg_requires_append(self, three_cluster_image):
        with pytest.raises(TypeError):
            kmeans_ndarray(three_cluster_image, nclusters=3, frames=object())

    def test_invalid_distance_raises(self, three_cluster_image):
        with pytest.raises(ValueError):
            kmeans_ndarray(three_cluster_image, nclusters=3,
                           distance='bogus')

    def test_unsupported_kwarg_raises(self, three_cluster_image):
        with pytest.raises(NameError):
            kmeans_ndarray(three_cluster_image, nclusters=3, bogus=True)

    def test_nan_data_raises(self, three_cluster_image):
        data = three_cluster_image.copy()
        data[0, 0, 0] = np.nan
        with pytest.raises(NaNValueError):
            kmeans_ndarray(data, nclusters=3)


class TestKmeansDispatch:
    '''Tests that `kmeans` correctly dispatches ndarray input.'''

    def test_ndarray_input_matches_kmeans_ndarray(self, three_cluster_image):
        kwargs = dict(nclusters=3, max_iterations=7, distance='L1')
        (class_map1, centers1) = spy.kmeans(three_cluster_image, **kwargs)
        (class_map2, centers2) = kmeans_ndarray(three_cluster_image,
                                                **kwargs)
        assert_array_equal(class_map1, class_map2)
        assert_allclose(centers1, centers2)


class TestKmeansLoopBranch:
    '''Tests the pure Python loop implementation used by `kmeans` for
    non-ndarray (e.g., SpyFile-like) input.'''

    @pytest.fixture
    def wrapped_image(self, three_cluster_image):
        return NonNdarrayImage(three_cluster_image)

    def test_converges_to_expected_clusters(self, wrapped_image):
        (class_map, centers) = spy.kmeans(wrapped_image, nclusters=3,
                                          max_iterations=20)
        assert_valid_clustering(class_map, centers)

    @pytest.mark.parametrize('distance', ['L1', 'L2', spy.L1, spy.L2])
    def test_distance_options(self, wrapped_image, distance):
        (class_map, centers) = spy.kmeans(wrapped_image, nclusters=3,
                                          distance=distance)
        assert_valid_clustering(class_map, centers)

    def test_start_clusters_wrong_shape_raises(self, wrapped_image):
        bad_start = np.zeros((2, wrapped_image.shape[2]))
        with pytest.raises(AssertionError):
            spy.kmeans(wrapped_image, nclusters=3, start_clusters=bad_start)

    def test_compare_kwarg_terminates_iteration(self, wrapped_image):
        calls = []

        def compare(old_clusters, clusters):
            calls.append(1)
            return True

        spy.kmeans(wrapped_image, nclusters=3, max_iterations=20,
                  compare=compare)
        assert len(calls) == 1

    def test_frames_kwarg_records_each_iteration(self, wrapped_image):
        frames = []
        (class_map, centers) = spy.kmeans(wrapped_image, nclusters=3,
                                          max_iterations=20, frames=frames)
        assert len(frames) >= 1
        assert_array_equal(frames[-1], class_map)

    def test_frames_kwarg_requires_append(self, wrapped_image):
        with pytest.raises(TypeError):
            spy.kmeans(wrapped_image, nclusters=3, frames=object())

    def test_invalid_distance_raises(self, wrapped_image):
        with pytest.raises(ValueError):
            spy.kmeans(wrapped_image, nclusters=3, distance='bogus')

    def test_unsupported_kwarg_raises(self, wrapped_image):
        with pytest.raises(NameError):
            spy.kmeans(wrapped_image, nclusters=3, bogus=True)

    def test_keyboard_interrupt_returns_previous_iteration(self,
                                                           wrapped_image,
                                                           monkeypatch):
        '''A KeyboardInterrupt raised while assigning pixels during the
        second iteration should cause `kmeans` to return the clustering
        from the (already fully-converged, in this synthetic case) first
        iteration rather than propagating the exception.'''
        nrows = wrapped_image.shape[0]
        original = spy._status.update_percentage
        calls = {'n': 0}

        def maybe_raise(*args, **kwargs):
            calls['n'] += 1
            if calls['n'] > nrows:
                raise KeyboardInterrupt
            return original(*args, **kwargs)

        monkeypatch.setattr(spy._status, 'update_percentage', maybe_raise)
        (class_map, centers) = spy.kmeans(wrapped_image, nclusters=3,
                                          max_iterations=20)
        assert_valid_clustering(class_map, centers)
