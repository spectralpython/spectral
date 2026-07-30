'''Tests for spectral.graphics.ndwindow (the N-dimensional feature display).

Widget construction and interaction logic (keyboard/mouse handling, octant
feature assignment) run in any environment with PySide6 installed -- Qt's
"offscreen" platform plugin is enough, no real or virtual display required.
Note that `NDWindow.set_data` (which normalizes point data and computes
RGBA pixel-ID color masks) issues real OpenGL calls and is only invoked by
`initializeGL`, i.e. by an actual paint pass -- so any test that depends on
it having run needs the `require_gui3d` fixture and a real Show() + event
pump, same as the rendering tests in test_hypercube.py.

An exception raised from inside initializeGL/paintGL was found to segfault
the interpreter rather than propagate as a normal Python exception, so the
rendering tests here stick to valid, well-formed input.

To run just this file:

    # pytest spectral/tests/test_ndwindow.py
'''

import warnings

import numpy as np
import pytest

from spectral.tests.conftest import import_or_require

import_or_require('PySide6')
import_or_require('OpenGL')

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from spectral.graphics.ndwindow import (NDWindow, create_mirrored_octants,
                                        random_subset, rtp_to_xyz,
                                        validate_args, xyz_to_rtp)

pytestmark = pytest.mark.gui3d


def make_key_event(key, text, modifiers=Qt.KeyboardModifier.NoModifier):
    return QKeyEvent(QEvent.Type.KeyPress, key, modifiers, text=text)


def make_mouse_event(pos, event_type=QEvent.Type.MouseMove,
                     modifiers=Qt.KeyboardModifier.NoModifier):
    with warnings.catch_warnings():
        # The 5-arg QMouseEvent constructor is deprecated in favor of one
        # that also takes a QPointingDevice; harmless for these tests.
        warnings.simplefilter('ignore', DeprecationWarning)
        return QMouseEvent(event_type, QPointF(*pos), Qt.MouseButton.LeftButton,
                           Qt.MouseButton.LeftButton, modifiers)


class TestCoordinateConversion:
    def test_round_trip(self):
        (r, theta, phi) = (5.0, 45.0, 30.0)
        xyz = rtp_to_xyz(r, theta, phi)
        assert xyz_to_rtp(*xyz) == pytest.approx([r, theta, phi])


class TestCreateMirroredOctants:
    def test_maps_six_semi_axis_features_to_eight_octant_triplets(self):
        octants = create_mirrored_octants([0, 1, 2, 3, 4, 5])
        assert len(octants) == 8
        assert octants[0] == [0, 1, 2]
        # The +x/+y/+z octant and its mirror across x share the y, z
        # features (indices 1 and 2), differing only in the x feature.
        assert octants[1] == [3, 1, 2]


class TestRandomSubset:
    def test_returns_requested_unique_count(self):
        result = random_subset(list(range(10)), 4)
        assert len(result) == 4
        assert len(set(result)) == 4
        assert all(0 <= x < 10 for x in result)

    def test_raises_if_sequence_too_small(self):
        with pytest.raises(Exception):
            random_subset([0, 1], 3)


class TestValidateArgs:
    def test_requires_ndarray(self):
        with pytest.raises(TypeError):
            validate_args([[1, 2, 3]])

    def test_requires_three_dimensions(self):
        with pytest.raises(ValueError):
            validate_args(np.zeros((3, 3)))

    def test_requires_at_least_three_features(self):
        with pytest.raises(ValueError):
            validate_args(np.zeros((3, 3, 2)))

    def test_accepts_valid_data(self):
        validate_args(np.zeros((3, 3, 6)))  # Should not raise.

    def test_classes_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            validate_args(np.zeros((3, 3, 6)), classes=np.zeros((2, 2)))

    def test_feature_index_out_of_range_raises(self):
        with pytest.raises(ValueError):
            validate_args(np.zeros((3, 3, 6)), features=[0, 1, 99])

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            validate_args(np.zeros((3, 3, 6)), size=(10, 0))


@pytest.fixture
def nd_data():
    return np.random.rand(6, 6, 8).astype(np.float32)


@pytest.fixture
def window(qapp, nd_data):
    classes = np.zeros((6, 6), int)
    classes[0:2, 0:2] = 1
    w = NDWindow(nd_data, None, -1, size=(64, 64), classes=classes,
                features=list(range(6)))
    yield w
    w.close()


class TestConstruction:
    '''Tests NDWindow's initial state, without needing real GL.

    `quadrant_mode` starts as None: it's only set once `set_data` runs
    (from `initializeGL`, i.e. an actual paint pass -- see TestRendering).
    '''

    def test_defaults(self, window):
        assert window.win_size == (64, 64)
        assert window.windowTitle() == 'ND Window'
        assert window.quadrant_mode is None
        assert window.camera_pos_rtp == [7.0, 45.0, 30.0]
        assert window.show_axes_tf is True
        assert window.point_size == 1.0
        assert window.mouse_panning is False

    def test_max_menu_class_derived_from_classes(self, window):
        assert window.max_menu_class == 2  # max(classes) == 1, plus one


class TestOctantDisplayFeatures:
    '''Tests set_octant_display_features and randomize_features, both pure
    Python state manipulation with no OpenGL calls.'''

    def test_three_features_is_single_octant_mode(self, window):
        window.set_octant_display_features([0, 1, 2])
        assert window.quadrant_mode == 'single'
        assert window.octant_features[0] == [0, 1, 2]
        assert window.octant_features[1:] == [None] * 7
        assert list(window.target_pos) == [0.5, 0.5, 0.5]

    def test_six_features_is_mirrored_octant_mode(self, window):
        window.set_octant_display_features([0, 1, 2, 3, 4, 5])
        assert window.quadrant_mode == 'mirrored'
        assert len(window.octant_features) == 8
        assert list(window.target_pos) == [0.0, 0.0, 0.0]

    def test_eight_triplets_is_independent_octant_mode(self, window):
        window.set_octant_display_features([[0, 1, 2]] * 8)
        assert window.quadrant_mode == 'independent'

    def test_none_defaults_to_six_features(self, window):
        window.set_octant_display_features(None)
        assert window.quadrant_mode == 'mirrored'

    def test_randomize_features_matches_current_mode(self, window):
        window.quadrant_mode = 'single'
        window.randomize_features()
        assert window.quadrant_mode == 'single'
        assert len(window.octant_features[0]) == 3

        window.quadrant_mode = 'mirrored'
        window.randomize_features()
        assert window.quadrant_mode == 'mirrored'

    def test_reset_view_geometry(self, window):
        window.target_pos = np.array([1.0, 1.0, 1.0])
        window.camera_pos_rtp = [1.0, 1.0, 1.0]
        window.reset_view_geometry()
        assert list(window.target_pos) == [0.0, 0.0, 0.0]
        assert window.camera_pos_rtp == [2.5, 45.0, 30.0]


class TestSetFeatures:
    '''Tests the public set_features validation (used by NDWindowProxy).'''

    def test_single_mode_requires_three_features(self, window):
        with pytest.raises(Exception):
            window.set_features([0, 1], mode='single')
        window.set_features([0, 1, 2], mode='single')
        assert window.quadrant_mode == 'single'

    def test_mirrored_mode_requires_six_features(self, window):
        with pytest.raises(Exception):
            window.set_features([0, 1, 2], mode='mirrored')
        window.set_features([0, 1, 2, 3, 4, 5], mode='mirrored')
        assert window.quadrant_mode == 'mirrored'

    def test_unrecognized_mode_raises(self, window):
        with pytest.raises(Exception):
            window.set_features([0, 1, 2], mode='bogus')


class TestKeyboardHandling:
    '''Tests NDWindow.keyPressEvent, per its documented keybinds.'''

    def test_a_toggles_axis_display(self, window):
        assert window.show_axes_tf is True
        window.keyPressEvent(make_key_event(Qt.Key.Key_A, 'a'))
        assert window.show_axes_tf is False

    def test_m_toggles_mouse_panning(self, window):
        assert window.mouse_panning is False
        window.keyPressEvent(make_key_event(Qt.Key.Key_M, 'm'))
        assert window.mouse_panning is True

    def test_p_increases_point_size_uppercase_decreases_with_floor(self, window):
        window.keyPressEvent(make_key_event(Qt.Key.Key_P, 'p'))
        assert window.point_size == 2.0
        shift = Qt.KeyboardModifier.ShiftModifier
        window.keyPressEvent(make_key_event(Qt.Key.Key_P, 'P', modifiers=shift))
        assert window.point_size == 1.0
        window.keyPressEvent(make_key_event(Qt.Key.Key_P, 'P', modifiers=shift))
        assert window.point_size == 1.0  # Floored at 1.0.

    def test_u_and_shift_u_toggle_unassigned_and_assigned_visibility(self, window):
        assert window._show_unassigned is True
        window.keyPressEvent(make_key_event(Qt.Key.Key_U, 'u'))
        assert window._show_unassigned is False
        shift = Qt.KeyboardModifier.ShiftModifier
        assert window._show_assigned is True
        window.keyPressEvent(make_key_event(Qt.Key.Key_U, 'U', modifiers=shift))
        assert window._show_assigned is False

    def test_r_resets_view_geometry(self, window):
        window.camera_pos_rtp = [1.0, 1.0, 1.0]
        window.keyPressEvent(make_key_event(Qt.Key.Key_R, 'r'))
        assert window.camera_pos_rtp == [2.5, 45.0, 30.0]

    def test_d_cycles_quadrant_mode_when_enough_features(self, window):
        # `window`'s data has 8 features, so all three modes are reachable.
        window.quadrant_mode = 'single'
        window.keyPressEvent(make_key_event(Qt.Key.Key_D, 'd'))
        assert window.quadrant_mode == 'mirrored'
        window.keyPressEvent(make_key_event(Qt.Key.Key_D, 'd'))
        assert window.quadrant_mode == 'independent'
        window.keyPressEvent(make_key_event(Qt.Key.Key_D, 'd'))
        assert window.quadrant_mode == 'single'

    def test_d_is_a_noop_with_fewer_than_six_features(self, qapp):
        data = np.random.rand(6, 6, 4).astype(np.float32)
        w = NDWindow(data, None, -1, size=(64, 64),
                    classes=np.zeros((6, 6), int), features=[0, 1, 2])
        try:
            w.quadrant_mode = 'single'
            w.keyPressEvent(make_key_event(Qt.Key.Key_D, 'd'))
            assert w.quadrant_mode == 'single'
        finally:
            w.close()

    def test_q_closes_window(self, window):
        window.Show()
        assert window.isVisible()
        window.keyPressEvent(make_key_event(Qt.Key.Key_Q, 'q'))
        assert not window.isVisible()

    def test_c_key_does_not_raise(self, window):
        import_or_require('matplotlib')
        window.keyPressEvent(make_key_event(Qt.Key.Key_C, 'c'))

    def test_view_class_image_returns_linked_image_view(self, window):
        import_or_require('matplotlib')
        view = window.view_class_image()
        try:
            assert view.is_shown
            assert view.callbacks_common is window.callbacks
        finally:
            import matplotlib.pyplot as plt
            plt.close(view.axes.figure)


class TestMouseHandler:
    '''Tests MouseHandler's DEFAULT/BOX_SELECT/ZOOMING mode machine.'''

    def test_plain_drag_rotates_camera(self, window):
        handler = window.mouse_handler
        handler.left_down(make_mouse_event((10, 10),
                                           event_type=QEvent.Type.MouseButtonPress))
        start = list(window.camera_pos_rtp)
        handler.motion(make_mouse_event((40, 10)))
        assert window.camera_pos_rtp[2] != start[2]
        assert handler.mode == 'DEFAULT'

    def test_drag_pans_target_when_mouse_panning_enabled(self, window):
        window.mouse_panning = True
        handler = window.mouse_handler
        handler.left_down(make_mouse_event((10, 10),
                                           event_type=QEvent.Type.MouseButtonPress))
        start_target = list(window.target_pos)
        handler.motion(make_mouse_event((40, 10)))
        assert list(window.target_pos) != start_target

    def test_ctrl_drag_zooms(self, window):
        handler = window.mouse_handler
        shift = Qt.KeyboardModifier.ControlModifier
        handler.left_down(make_mouse_event(
            (10, 10), event_type=QEvent.Type.MouseButtonPress, modifiers=shift))
        assert handler.mode == 'ZOOMING'
        start_r = window.camera_pos_rtp[0]
        handler.motion(make_mouse_event((40, 10), modifiers=shift))
        assert window.camera_pos_rtp[0] != start_r
        handler.left_up(make_mouse_event((40, 10),
                                         event_type=QEvent.Type.MouseButtonRelease))
        assert handler.mode == 'DEFAULT'

    def test_shift_drag_selects_box_and_confirms_on_shift_release(self, window):
        handler = window.mouse_handler
        shift = Qt.KeyboardModifier.ShiftModifier
        handler.left_down(make_mouse_event(
            (5, 5), event_type=QEvent.Type.MouseButtonPress, modifiers=shift))
        assert handler.mode == 'BOX_SELECT'
        handler.motion(make_mouse_event((20, 20), modifiers=shift))
        assert window._selection_box is not None
        handler.left_up(make_mouse_event(
            (20, 20), event_type=QEvent.Type.MouseButtonRelease, modifiers=shift))
        assert handler.mode == 'DEFAULT'
        assert window._selection_box is not None

    def test_releasing_shift_before_mouse_up_cancels_selection(self, window):
        handler = window.mouse_handler
        shift = Qt.KeyboardModifier.ShiftModifier
        handler.left_down(make_mouse_event(
            (5, 5), event_type=QEvent.Type.MouseButtonPress, modifiers=shift))
        handler.motion(make_mouse_event((20, 20), modifiers=shift))
        handler.left_up(make_mouse_event((20, 20),
                                         event_type=QEvent.Type.MouseButtonRelease))
        assert window._selection_box is None
        # NOTE: actual (arguably surprising) behavior -- the handler's mode
        # is not reset to 'DEFAULT' on a cancelled box selection, unlike the
        # confirmed-selection and zoom-release paths above.
        assert handler.mode == 'BOX_SELECT'

    def test_ctrl_shift_click_queues_pixel_info_command(self, window):
        handler = window.mouse_handler
        modifiers = (Qt.KeyboardModifier.ControlModifier |
                    Qt.KeyboardModifier.ShiftModifier)
        assert len(window._display_commands) == 0
        handler.left_down(make_mouse_event(
            (10, 10), event_type=QEvent.Type.MouseButtonPress, modifiers=modifiers))
        assert len(window._display_commands) == 1
        assert handler.mode == 'DEFAULT'

    def test_motion_without_button_down_is_ignored(self, window):
        handler = window.mouse_handler
        start = list(window.camera_pos_rtp)
        handler.motion(make_mouse_event((40, 40)))
        assert window.camera_pos_rtp == start


class TestRendering:
    '''Tests that actually require a working OpenGL surface to render into.'''

    def test_show_initializes_gl_state_and_normalizes_data(self, require_gui3d,
                                                            window, qapp):
        window.Show()
        for _ in range(10):
            qapp.processEvents()
        assert isinstance(window.gllist_id, int)
        # `set_data` (only reachable via a real paint pass) normalizes point
        # data to fill the display octant(s).
        assert window.data.min() >= 0.0
        assert window.data.max() <= 1.0
        assert window.quadrant_mode == 'mirrored'  # features=range(6)
