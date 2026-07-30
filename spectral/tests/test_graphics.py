'''Tests for spectral.graphics.graphics.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_graphics.py
'''

import os

import numpy as np
from numpy.testing import assert_allclose
import pytest

from spectral.graphics.colorscale import create_default_color_scale
from spectral.graphics.graphics import (get_rgb, get_rgb_meta, make_pil_image,
                                        save_rgb, running_ipython,
                                        warn_no_ipython)
from spectral.image import Image
from spectral.tests.conftest import import_or_require


class FakeImage(Image):
    '''A minimal spectral.Image stand-in, without needing a real SpyFile.'''

    def __init__(self, data, metadata=None):
        self.data = np.asarray(data, dtype=float)
        self.shape = self.data.shape
        self.metadata = metadata or {}

    def read_bands(self, bands):
        return self.data[:, :, bands]


class TestGetRgbBandSelection:
    '''Tests how get_rgb_meta picks which bands to display.'''

    def test_monochrome_ndarray_replicates_band_to_rgb(self):
        data = np.array([[0., 10.], [20., 30.]])
        rgb, meta = get_rgb_meta(data)
        assert meta['mode'] == 'monochrome'
        assert meta['bands'] == [0]
        assert rgb.shape == (2, 2, 3)
        # Fully stretched (default stretch spans the full data range).
        assert_allclose(rgb[0, 0], [0, 0, 0])
        assert_allclose(rgb[1, 1], [1, 1, 1])

    def test_three_band_ndarray_used_as_is(self):
        data = np.random.rand(4, 4, 3)
        rgb, meta = get_rgb_meta(data, stretch=(0.0, 1.0))
        assert meta['mode'] == 'rgb'
        assert meta['bands'] == [0, 1, 2]

    def test_more_than_three_bands_picks_first_middle_last(self):
        data = np.arange(4 * 4 * 5).reshape(4, 4, 5).astype(float)
        _, meta = get_rgb_meta(data)
        assert meta['bands'] == [0, 5 / 2, 4]

    def test_explicit_bands_kwarg_used(self):
        data = np.arange(4 * 4 * 5).reshape(4, 4, 5).astype(float)
        _, meta = get_rgb_meta(data, bands=[4, 0, 1])
        assert meta['bands'] == [4, 0, 1]

    def test_invalid_number_of_bands_raises(self):
        data = np.arange(4 * 4 * 5).reshape(4, 4, 5).astype(float)
        with pytest.raises(Exception):
            get_rgb(data, bands=[0, 1])

    def test_image_default_bands_metadata_used(self):
        data = np.arange(4 * 4 * 5).reshape(4, 4, 5).astype(float)
        img = FakeImage(data, metadata={'default bands': ['2', '1', '0']})
        _, meta = get_rgb_meta(img)
        assert meta['bands'] == [2, 1, 0]

    def test_image_unparseable_default_bands_falls_back_with_warning(self):
        data = np.arange(4 * 4 * 5).reshape(4, 4, 5).astype(float)
        img = FakeImage(data, metadata={'default bands': ['x', 'y', 'z']})
        with pytest.warns(UserWarning):
            _, meta = get_rgb_meta(img)
        assert meta['bands'] == [0, 2, 4]

    def test_image_single_band_defaults_to_band_zero(self):
        data = np.arange(4 * 4).reshape(4, 4, 1).astype(float)
        img = FakeImage(data)
        _, meta = get_rgb_meta(img)
        assert meta['bands'] == [0]
        assert meta['mode'] == 'monochrome'

    def test_ndarray_single_band_nonzero_index_raises(self):
        data = np.arange(4 * 4).reshape(4, 4, 1).astype(float)
        with pytest.raises(ValueError):
            get_rgb(data, bands=[1])


class TestGetRgbColorModes:
    '''Tests the indexed, color-scaled, and masked display paths.'''

    def test_indexed_colors(self):
        idx = np.array([[0, 1], [1, 0]])
        colors = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]])
        rgb, meta = get_rgb_meta(idx, colors=colors)
        assert meta['mode'] == 'indexed'
        assert_allclose(rgb[0, 0], [0, 0, 0])
        assert_allclose(rgb[0, 1], [1, 0, 0])

    def test_color_scale_with_auto_scale(self):
        scale = create_default_color_scale()
        data = np.array([[0., 25.], [50., 10.]])
        rgb, meta = get_rgb_meta(data, color_scale=scale, auto_scale=True)
        assert meta['mode'] == 'scaled'
        assert rgb.shape == (2, 2, 3)

    def test_mask_fills_background_color(self):
        data = np.array([[0., 10.], [20., 30.]])
        mask = np.array([[1, 0], [1, 1]])
        rgb, _ = get_rgb_meta(data, mask=mask, bg=(9, 9, 9))
        assert_allclose(rgb[0, 1], np.array([9, 9, 9]) / 255.)


class TestGetRgbStretchAndBounds:
    '''Tests stretch/bounds keyword handling and validation.'''

    def test_invalid_keyword_raises(self):
        data = np.random.rand(4, 4, 3)
        with pytest.raises(ValueError):
            get_rgb(data, foo=1)

    def test_bounds_invalid_shape_raises(self):
        data = np.random.rand(4, 4, 3)
        with pytest.raises(ValueError):
            get_rgb(data, bounds=[1, 2, 3])

    def test_stretch_numeric_out_of_range_raises(self):
        data = np.random.rand(4, 4, 3)
        with pytest.raises(ValueError):
            get_rgb(data, stretch=1.5)

    def test_boolean_stretch_is_deprecated(self):
        data = np.random.rand(4, 4, 3)
        with pytest.warns(UserWarning):
            get_rgb(data, stretch=True)

    def test_explicit_bounds_clips_values(self):
        data = np.array([[0., 5.], [10., 20.]])
        rgb, _ = get_rgb_meta(data, bounds=(0, 10))
        assert_allclose(rgb[0, 0], [0, 0, 0])
        assert_allclose(rgb[1, 0], [1, 1, 1])
        # Value above the upper bound is clipped to 1.
        assert_allclose(rgb[1, 1], [1, 1, 1])


class TestSaveAndMakePilImage:
    '''Tests save_rgb and make_pil_image, which wrap PIL for file output.'''

    @pytest.fixture(autouse=True)
    def _require_pil(self):
        import_or_require('PIL')

    def test_make_pil_image_returns_correct_size(self):
        data = np.random.rand(5, 6, 3)
        img = make_pil_image(data)
        assert img.size == (6, 5)

    def test_save_rgb_writes_readable_file(self, testdir):
        from PIL import Image as PILImage

        data = np.random.rand(5, 6, 3)
        path = os.path.join(testdir, 'out.png')
        save_rgb(path, data)
        with PILImage.open(path) as im:
            assert im.size == (6, 5)
            assert im.mode == 'RGB'

    def test_save_rgb_indexed_colors(self, testdir):
        from PIL import Image as PILImage

        idx = np.array([[0, 1], [1, 0]])
        colors = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]])
        path = os.path.join(testdir, 'indexed.png')
        save_rgb(path, idx, colors=colors)
        with PILImage.open(path) as im:
            assert im.size == (2, 2)


class TestIPythonHelpers:
    '''Tests running_ipython and warn_no_ipython.'''

    def test_running_ipython_false_outside_ipython(self):
        assert running_ipython() is False

    def test_warn_no_ipython_issues_user_warning(self):
        with pytest.warns(UserWarning):
            warn_no_ipython()
