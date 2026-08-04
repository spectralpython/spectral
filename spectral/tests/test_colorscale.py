'''Tests for spectral.graphics.colorscale.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_colorscale.py
'''

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from spectral.graphics.colorscale import ColorScale, create_default_color_scale

pytestmark = pytest.mark.graphics


class TestColorScaleConstruction:
    '''Tests validation and setup performed by ColorScale.__init__.'''

    def test_invalid_colors_shape_raises(self):
        # `colors` must be an (N, 3) array. The implementation raises a bare
        # string, which Python 3 turns into a TypeError.
        with pytest.raises(TypeError):
            ColorScale([0, 1], np.zeros((2, 2)))

    def test_mismatched_levels_and_colors_raises(self):
        with pytest.raises(TypeError):
            ColorScale([0, 1, 2], np.zeros((2, 3)))

    def test_num_tics_below_two_raises(self):
        with pytest.raises(ValueError):
            ColorScale([0, 1], np.zeros((2, 3)), num_tics=1)

    def test_default_num_tics_matches_color_count(self):
        colors = np.array([[0, 0, 0], [255, 255, 255]])
        scale = ColorScale([0, 10], colors)
        assert scale.size == len(colors)

    def test_levels_converted_to_float(self):
        colors = np.array([[0, 0, 0], [255, 255, 255]])
        scale = ColorScale([0, 10], colors)
        assert scale.min == 0.0
        assert scale.max == 10.0
        assert scale.span == 10.0


class TestColorScaleCall:
    '''Tests the __call__ operator that maps a scalar to an RGB color.'''

    @pytest.fixture(autouse=True)
    def setup(self):
        self.scale = create_default_color_scale()

    def test_value_below_min_returns_background_color(self):
        assert_array_equal(self.scale(-5), self.scale.bgColor)

    def test_custom_background_color_used_below_min(self):
        self.scale.set_background_color((9, 8, 7))
        assert_array_equal(self.scale(-1), [9, 8, 7])

    def test_value_at_or_above_max_returns_last_color_tic(self):
        assert_array_equal(self.scale(50), self.scale.colorTics[-1])
        assert_array_equal(self.scale(1000), self.scale.colorTics[-1])

    def test_value_at_tic_level_returns_expected_color(self):
        # The default scale's tics align exactly with the 6 base colors
        # (black, blue, green, red, yellow, white), with no interpolation.
        assert_array_equal(self.scale(0), [0, 0, 0])
        assert_array_equal(self.scale(10), [0, 0, 255])


class TestColorScaleMutators:
    '''Tests set_background_color and set_range.'''

    @pytest.fixture(autouse=True)
    def setup(self):
        self.scale = create_default_color_scale()

    def test_set_background_color(self):
        self.scale.set_background_color([1, 2, 3])
        assert_array_equal(self.scale.bgColor, [1, 2, 3])

    def test_set_background_color_invalid_shape_raises(self):
        with pytest.raises(TypeError):
            self.scale.set_background_color([1, 2])

    def test_set_range(self):
        self.scale.set_range(100, 200)
        assert self.scale.min == 100
        assert self.scale.max == 200
        assert self.scale.span == 100


class TestCreateDefaultColorScale:
    '''Tests the create_default_color_scale factory function.'''

    def test_default_has_six_tics_no_interpolation(self):
        scale = create_default_color_scale()
        assert scale.size == 6
        assert_array_equal(scale.tics, [0, 10, 20, 30, 40, 50])

    def test_ntics_less_than_base_colors_raises(self):
        with pytest.raises(ValueError):
            create_default_color_scale(5)

    def test_ntics_interpolates_additional_colors(self):
        scale = create_default_color_scale(12)
        assert scale.size == 12
        assert_array_equal(scale.colorTics[0], [0, 0, 0])
        assert_array_equal(scale.colorTics[-1], [255, 255, 255])
