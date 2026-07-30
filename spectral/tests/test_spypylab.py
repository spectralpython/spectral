'''Tests for spectral.graphics.spypylab.

These tests exercise the matplotlib-based 2D image display (`ImageView`,
`imshow`, `plot`) headlessly using the Agg backend (forced in conftest.py),
so no display or window manager is required.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_spypylab.py
'''

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from spectral.tests.conftest import import_or_require

import_or_require('matplotlib')

from spectral.graphics.spypylab import (ImageView, KeyParser, MplCallback,
                                        imshow, plot, rowcol_to_xy,
                                        xy_to_rowcol)


class FakeKeyEvent:
    '''Stands in for a matplotlib KeyEvent for direct handler calls.'''

    def __init__(self, key):
        self.key = key


class FakeMouseEvent:
    '''Stands in for a matplotlib MouseEvent for direct handler calls.'''

    def __init__(self, inaxes, xdata=0., ydata=0., button=1,
                 dblclick=False, key=None):
        self.inaxes = inaxes
        self.xdata = xdata
        self.ydata = ydata
        self.button = button
        self.dblclick = dblclick
        self.key = key


class TestCoordinateConversion:
    '''Tests xy_to_rowcol and rowcol_to_xy.'''

    def test_xy_to_rowcol_rounds_to_nearest_pixel(self):
        assert xy_to_rowcol(2.4, 3.6) == (4, 2)

    def test_rowcol_to_xy_returns_pixel_center(self):
        assert rowcol_to_xy(3, 2) == (2.0, 3.0)

    def test_round_trip(self):
        (x, y) = rowcol_to_xy(5, 7)
        assert xy_to_rowcol(x, y) == (5, 7)


class TestKeyParser:
    '''Tests the KeyParser helper used to interpret matplotlib key strings.'''

    def test_none_key(self):
        kp = KeyParser(None)
        assert kp.key is None
        assert kp.modifiers == set()

    def test_plain_key(self):
        kp = KeyParser('a')
        assert kp.key == 'a'
        assert kp.modifiers == set()

    def test_single_modifier(self):
        kp = KeyParser('ctrl+a')
        assert kp.key == 'a'
        assert kp.has_mod('ctrl')
        assert not kp.has_mod('shift')

    def test_modifier_aliases(self):
        # "control" is an alias for "ctrl".
        kp = KeyParser('control+a')
        assert kp.has_mod('ctrl')

    def test_multiple_modifiers(self):
        kp = KeyParser('ctrl+alt+shift+z')
        assert kp.key == 'z'
        assert kp.modifiers == {'ctrl', 'alt', 'shift'}

    def test_modifiers_only_no_key(self):
        # If every token is a modifier, `key` remains None.
        kp = KeyParser('ctrl+shift')
        assert kp.key is None
        assert kp.modifiers == {'ctrl', 'shift'}

    def test_unrecognized_modifier_raises(self):
        with pytest.raises(ValueError):
            KeyParser('foo+a')

    def test_mods_are_is_a_subset_check(self):
        # Despite the docstring, `mods_are` only checks that the given
        # modifiers are present, not that they are the *only* ones present.
        kp = KeyParser('ctrl+shift+a')
        assert kp.mods_are('ctrl')
        assert kp.mods_are('ctrl', 'shift')
        assert not kp.mods_are('alt')


class TestMplCallback:
    '''Tests MplCallback connect/disconnect and exception handling.'''

    def test_connect_receives_events(self):
        from matplotlib.cbook import CallbackRegistry
        registry = CallbackRegistry()
        calls = []
        cb = MplCallback(registry=registry, event='ev',
                         callback=lambda *a: calls.append(a))
        cb.connect()
        registry.process('ev', 'payload')
        assert calls == [('payload',)]
        assert cb.is_connected

    def test_disconnect_stops_events(self):
        from matplotlib.cbook import CallbackRegistry
        registry = CallbackRegistry()
        calls = []
        cb = MplCallback(registry=registry, event='ev',
                         callback=lambda *a: calls.append(a))
        cb.connect()
        cb.disconnect()
        assert not cb.is_connected
        registry.process('ev', 'payload')
        assert calls == []

    def test_exception_in_callback_disconnects_silently_by_default(self):
        from matplotlib.cbook import CallbackRegistry
        registry = CallbackRegistry()

        def bad_callback(*a):
            raise RuntimeError('boom')

        cb = MplCallback(registry=registry, event='ev', callback=bad_callback)
        cb.connect()
        registry.process('ev', 'payload')  # Should not raise.
        assert not cb.is_connected

    def test_exception_in_callback_reraised_when_requested(self):
        from matplotlib.cbook import CallbackRegistry
        registry = CallbackRegistry()

        def bad_callback(*a):
            raise RuntimeError('boom')

        cb = MplCallback(registry=registry, event='ev', callback=bad_callback)
        cb.raise_event_exceptions = True
        cb.connect()
        # Call the callback directly rather than via registry.process():
        # CallbackRegistry.process() has its own exception handler
        # (matplotlib.cbook._exception_printer) that only lets exceptions
        # propagate when no interactive GUI framework is detected as
        # running -- which depends on ambient process state (e.g. whether
        # a QApplication exists) unrelated to what's being tested here,
        # namely MplCallback's own re-raise behavior.
        with pytest.raises(RuntimeError):
            cb('payload')


class TestImageViewShow:
    '''Tests basic construction and display of an ImageView.'''

    def test_show_creates_data_axes(self):
        data = np.random.rand(6, 8, 3)
        view = ImageView(data=data)
        assert not view.is_shown
        view.show()
        assert view.is_shown
        assert view.display_mode == 'data'
        assert view.data_axes is not None
        assert view.axes is not None

    def test_show_twice_warns_and_is_noop(self):
        data = np.random.rand(4, 4, 3)
        view = ImageView(data=data)
        view.show()
        with pytest.warns(UserWarning):
            view.show()

    def test_classes_only_guesses_classes_mode(self):
        classes = np.zeros((4, 4), int)
        view = ImageView(classes=classes)
        view.show()
        assert view.display_mode == 'classes'

    def test_show_with_no_data_raises(self):
        view = ImageView()
        with pytest.raises(Exception):
            view.show()


class TestImageViewDisplayModes:
    '''Tests set_display_mode transitions between data/classes/overlay.'''

    @pytest.fixture(autouse=True)
    def setup(self):
        data = np.random.rand(6, 8, 3)
        classes = np.zeros((6, 8), int)
        classes[0:3, 0:3] = 1
        self.view = ImageView(data=data, classes=classes)
        self.view.show()

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            self.view.set_display_mode('bogus')

    def test_classes_mode_hides_data_shows_classes_opaque(self):
        self.view.set_display_mode('classes')
        assert self.view.data_axes.get_visible() is False
        assert self.view.class_axes.get_visible() is True
        assert self.view.class_axes.get_alpha() == 1

    def test_overlay_mode_shows_both_with_partial_alpha(self):
        self.view.set_display_mode('overlay')
        assert self.view.data_axes.get_visible() is True
        assert self.view.class_axes.get_visible() is True
        assert self.view.class_axes.get_alpha() == self.view.class_alpha

    def test_class_alpha_validation(self):
        with pytest.raises(ValueError):
            self.view.class_alpha = 1.5
        with pytest.raises(ValueError):
            self.view.class_alpha = -0.1
        self.view.class_alpha = 0.25
        assert self.view.class_alpha == 0.25

    def test_interpolation_setter_updates_axes(self):
        assert self.view.interpolation is None
        self.view.interpolation = 'nearest'
        assert self.view.data_axes.get_interpolation() == 'nearest'


class TestImageViewLabelRegion:
    '''Tests interactive pixel class labeling and its change notification.'''

    @pytest.fixture(autouse=True)
    def setup(self):
        data = np.random.rand(6, 8, 3)
        classes = np.zeros((6, 8), int)
        self.view = ImageView(data=data, classes=classes)
        self.view.show()
        self.events = []
        self.cb = MplCallback(registry=self.view.callbacks_common,
                              event='spy_classes_modified',
                              callback=lambda e: self.events.append(e))
        self.cb.connect()

    def test_label_region_updates_classes_and_fires_event(self):
        n = self.view.label_region([0, 3, 0, 2], 5)
        assert n == 6
        assert_array_equal(self.view.classes[0:3, 0:2], 5)
        assert_array_equal(self.view.classes[3:, :], 0)
        assert len(self.events) == 1
        assert self.events[0].nchanged == 6

    def test_relabeling_same_region_reassigns_zero_pixels(self):
        self.view.label_region([0, 3, 0, 2], 5)
        n = self.view.label_region([0, 3, 0, 2], 5)
        assert n == 0


class TestImageViewKeyboardHandler:
    '''Tests keyboard-driven interactions documented for ImageView.'''

    @pytest.fixture(autouse=True)
    def setup(self):
        data = np.random.rand(6, 8, 3)
        classes = np.zeros((6, 8), int)
        self.view = ImageView(data=data, classes=classes)
        self.view.show()
        self.handler = self.view.cb_keyboard

    def test_c_key_switches_to_classes_mode(self):
        self.handler.handle_event(FakeKeyEvent('c'))
        assert self.view.display_mode == 'classes'

    def test_capital_c_key_switches_to_overlay_mode(self):
        self.handler.handle_event(FakeKeyEvent('C'))
        assert self.view.display_mode == 'overlay'

    def test_d_key_switches_to_data_mode(self):
        self.view.set_display_mode('classes')
        self.handler.handle_event(FakeKeyEvent('d'))
        assert self.view.display_mode == 'data'

    def test_i_key_toggles_interpolation(self):
        self.handler.handle_event(FakeKeyEvent('i'))
        assert self.view.interpolation == 'nearest'
        self.handler.handle_event(FakeKeyEvent('i'))
        assert self.view.interpolation is None

    def test_alpha_keys_adjust_class_overlay_alpha(self):
        self.view.set_display_mode('overlay')
        start = self.view.class_alpha
        self.handler.handle_event(FakeKeyEvent('a'))
        assert self.view.class_alpha == pytest.approx(start - 0.05)
        self.handler.handle_event(FakeKeyEvent('A'))
        assert self.view.class_alpha == pytest.approx(start)

    def test_digit_enter_enter_assigns_class_to_selection(self):
        # Mirrors the documented workflow: shift-drag to select a region,
        # then type a numeric class ID and press ENTER twice to confirm.
        self.view.selection = [0, 3, 0, 2]
        self.handler.handle_event(FakeKeyEvent('1'))
        self.handler.handle_event(FakeKeyEvent('2'))
        assert self.handler.idstr == '12'
        self.handler.handle_event(FakeKeyEvent('enter'))
        assert self.handler.idstr == '12!'
        self.handler.handle_event(FakeKeyEvent('enter'))
        assert self.handler.idstr == ''
        assert_array_equal(self.view.classes[0:3, 0:2], 12)

    def test_enter_without_selection_does_not_assign(self):
        self.handler.handle_event(FakeKeyEvent('1'))
        self.handler.handle_event(FakeKeyEvent('enter'))
        self.handler.handle_event(FakeKeyEvent('enter'))
        assert not np.any(self.view.classes)

    def test_digit_ignored_without_selector(self):
        self.view.selector = None
        self.handler.handle_event(FakeKeyEvent('1'))
        assert self.handler.idstr == ''


class TestImageViewMouseHandler:
    '''Tests double-click-to-plot-spectrum behavior.'''

    def test_double_click_opens_spectrum_plot(self):
        import matplotlib.pyplot as plt

        data = np.random.rand(6, 8, 4)
        view = ImageView(data=data, source=data)
        view.show()
        nfigs_before = len(plt.get_fignums())

        event = FakeMouseEvent(view.axes, xdata=2.0, ydata=3.0, button=1,
                               dblclick=True)
        view.cb_mouse.handle_event(event)

        assert len(plt.get_fignums()) == nfigs_before + 1
        assert view.spectrum_plot_fig_id is not None

    def test_click_outside_axes_is_ignored(self):
        data = np.random.rand(6, 8, 3)
        view = ImageView(data=data, source=data)
        view.show()
        event = FakeMouseEvent(None, xdata=2.0, ydata=3.0, button=1,
                               dblclick=True)
        view.cb_mouse.handle_event(event)  # Should not raise.
        assert view.spectrum_plot_fig_id is None


class TestImageViewZoom:
    '''Tests open_zoom and pan_to.'''

    def test_open_zoom_creates_linked_view(self):
        data = np.random.rand(10, 10, 3)
        view = ImageView(data=data)
        view.show()
        zoom = view.open_zoom(center=(5, 5), size=4)
        assert zoom.is_shown
        assert zoom.interpolation == 'nearest'
        assert zoom.axes.get_xlim() == (3.0, 7.0)

    def test_pan_to_recenters_view(self):
        data = np.random.rand(10, 10, 3)
        view = ImageView(data=data)
        view.show()
        view.pan_to(2, 2)
        (xmin, xmax) = view.axes.get_xlim()
        assert (xmin + xmax) / 2 == pytest.approx(2)

    def test_pan_before_show_raises(self):
        data = np.random.rand(10, 10, 3)
        view = ImageView(data=data)
        with pytest.raises(Exception):
            view.pan_to(1, 1)


class TestImshowFunction:
    '''Tests the module-level imshow() convenience function.'''

    def test_returns_shown_view_with_title(self):
        data = np.random.rand(6, 8, 3)
        classes = np.zeros((6, 8), int)
        view = imshow(data, classes=classes, title='My Title')
        assert view.is_shown
        assert view.axes.get_title() == 'My Title'

    def test_data_over_three_bands_sets_source_automatically(self):
        data = np.random.rand(6, 8, 5)
        view = imshow(data)
        assert view.source is data


class TestPlotFunction:
    '''Tests the module-level plot() function.'''

    def test_plots_1d_data(self):
        import matplotlib.pyplot as plt
        plot(np.array([1., 2., 3., 4.]))
        lines = plt.gca().get_lines()
        assert len(lines) == 1
        assert_allclose(lines[0].get_ydata(), [1, 2, 3, 4])

    def test_plots_each_row_of_2d_data_as_a_series(self):
        import matplotlib.pyplot as plt
        data = np.array([[1., 2., 3.], [4., 5., 6.]])
        plot(data)
        lines = plt.gca().get_lines()
        assert len(lines) == 2

    def test_uses_band_centers_and_labels_axis(self):
        import matplotlib.pyplot as plt

        class FakeSource:
            class bands:
                centers = [500., 510., 520.]
                band_quantity = 'Wavelength'
                band_unit = 'nm'
            metadata = {}

        p = plot(np.array([1., 2., 3.]), source=FakeSource())
        assert_allclose(p[0].get_xdata(), [500., 510., 520.])
        assert plt.gca().get_xlabel() == 'Wavelength (nm)'

    def test_bad_bands_are_masked_as_nan(self):
        import matplotlib.pyplot as plt

        class FakeSource:
            class bands:
                centers = None
                band_quantity = None
                band_unit = None
            metadata = {'bbl': [1, 0, 1]}

        plot(np.array([1., 2., 3.]), source=FakeSource())
        ydata = plt.gca().get_lines()[0].get_ydata()
        assert np.isnan(ydata[1])
        assert_allclose(ydata[[0, 2]], [1, 3])
