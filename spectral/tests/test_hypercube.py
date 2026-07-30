'''Tests for spectral.graphics.hypercube (the 3D hypercube display).

Widget construction and interaction logic (keyboard/mouse handling) run in
any environment with PySide6 installed -- Qt's "offscreen" platform plugin
is enough, no real or virtual display required. Tests that need actual
OpenGL rendering (texture loading via Show()) additionally request the
`require_gui3d` fixture, which skips (or fails, see
--require-optional-deps=PySide6 in conftest.py) if this environment can't
actually render an OpenGL-backed Qt widget -- e.g. no display and no Xvfb.

Note: an exception raised from inside initializeGL/paintGL was found to
segfault the interpreter rather than propagate as a normal Python
exception, so the rendering tests here stick to valid, well-formed input.

To run just this file:

    # pytest spectral/tests/test_hypercube.py
'''

import warnings

import numpy as np
import pytest

from spectral.tests.conftest import import_or_require

import_or_require('PySide6')
import_or_require('OpenGL')

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from spectral.graphics.hypercube import HypercubeWindow, rtp_to_xyz, xyz_to_rtp

pytestmark = pytest.mark.gui3d


def make_key_event(key, text):
    return QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier,
                     text=text)


def make_mouse_event(pos, event_type=QEvent.Type.MouseMove,
                     modifiers=Qt.KeyboardModifier.NoModifier):
    with warnings.catch_warnings():
        # The 5-arg QMouseEvent constructor is deprecated in favor of one
        # that also takes a QPointingDevice; harmless for these tests.
        warnings.simplefilter('ignore', DeprecationWarning)
        return QMouseEvent(event_type, QPointF(*pos), Qt.MouseButton.LeftButton,
                           Qt.MouseButton.LeftButton, modifiers)


class TestCoordinateConversion:
    '''Tests the spherical/Cartesian conversion helpers.'''

    def test_round_trip(self):
        (r, theta, phi) = (5.0, 45.0, 30.0)
        xyz = rtp_to_xyz(r, theta, phi)
        result = xyz_to_rtp(*xyz)
        assert result == pytest.approx([r, theta, phi])


@pytest.fixture
def cube_data():
    return np.random.rand(8, 8, 5).astype(np.float32)


@pytest.fixture
def window(qapp, cube_data):
    w = HypercubeWindow(cube_data, None, -1, size=(64, 64))
    yield w
    w.close()


class TestConstruction:
    '''Tests HypercubeWindow's initial state, without needing real GL.'''

    def test_defaults(self, window):
        assert window.win_size == (64, 64)
        assert window.windowTitle() == 'Hypercube'
        assert window.clear_color == (0., 0., 0., 1.)
        assert window.cubeHeight == 1.0
        assert window.light is False
        assert window.camera_pos_rtp == [7.0, 45.0, 30.0]

    def test_custom_title_size_and_background(self, qapp, cube_data):
        w = HypercubeWindow(cube_data, None, -1, size=(80, 40),
                            title='My Cube', background=(1.0, 0.5, 0.0))
        try:
            assert w.win_size == (80, 40)
            assert w.windowTitle() == 'My Cube'
            assert w.clear_color == (1.0, 0.5, 0.0, 1.0)
        finally:
            w.close()

    def test_show_and_raise_do_not_raise(self, window):
        window.Show()
        assert window.isVisible()
        window.Raise()
        window.Show(False)
        assert not window.isVisible()


class TestKeyboardHandling:
    '''Tests HypercubeWindow.keyPressEvent, per its documented keybinds.'''

    def test_t_increases_cube_height(self, window):
        window.keyPressEvent(make_key_event(Qt.Key.Key_T, 't'))
        assert window.cubeHeight == pytest.approx(1.1)

    def test_g_decreases_cube_height(self, window):
        window.keyPressEvent(make_key_event(Qt.Key.Key_G, 'g'))
        assert window.cubeHeight == pytest.approx(0.9)

    def test_l_toggles_light(self, window):
        assert window.light is False
        window.keyPressEvent(make_key_event(Qt.Key.Key_L, 'l'))
        assert window.light is True
        window.keyPressEvent(make_key_event(Qt.Key.Key_L, 'l'))
        assert window.light is False

    def test_q_closes_window(self, window):
        window.Show()
        assert window.isVisible()
        window.keyPressEvent(make_key_event(Qt.Key.Key_Q, 'q'))
        assert not window.isVisible()


class TestMouseHandler:
    '''Tests MouseHandler's rotate/zoom/pan logic used by mouse drag events.'''

    def test_plain_drag_rotates_camera(self, window):
        handler = window.mouse_handler
        handler.left_down(make_mouse_event((10, 10),
                                           event_type=QEvent.Type.MouseButtonPress))
        start = list(window.camera_pos_rtp)
        handler.motion(make_mouse_event((40, 10)))
        assert window.camera_pos_rtp[0] == start[0]  # radius unchanged
        assert window.camera_pos_rtp[2] != start[2]  # phi (azimuth) changed

    def test_ctrl_drag_zooms(self, window):
        handler = window.mouse_handler
        handler.left_down(make_mouse_event((10, 10),
                                           event_type=QEvent.Type.MouseButtonPress))
        start_r = window.camera_pos_rtp[0]
        handler.motion(make_mouse_event(
            (40, 10), modifiers=Qt.KeyboardModifier.ControlModifier))
        assert window.camera_pos_rtp[0] != start_r

    def test_shift_drag_pans_target(self, window):
        handler = window.mouse_handler
        handler.left_down(make_mouse_event((10, 10),
                                           event_type=QEvent.Type.MouseButtonPress))
        start_target = list(window.target_pos)
        handler.motion(make_mouse_event(
            (10, 40), modifiers=Qt.KeyboardModifier.ShiftModifier))
        assert list(window.target_pos) != start_target

    def test_motion_without_button_down_is_ignored(self, window):
        handler = window.mouse_handler
        start = list(window.camera_pos_rtp)
        handler.motion(make_mouse_event((40, 40)))
        assert window.camera_pos_rtp == start


class TestRendering:
    '''Tests that actually require a working OpenGL surface to render into.'''

    def test_show_loads_six_cube_face_textures(self, require_gui3d, window, qapp):
        window.Show()
        for _ in range(10):
            qapp.processEvents()
        assert hasattr(window, 'textures')
        assert len(window.textures) == 6
