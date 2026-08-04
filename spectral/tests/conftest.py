'''Shared pytest fixtures for the SPy test suite.

Common fixtures:

    `testdir`
        A per-module temp directory for files written during tests. Using
        module scope (instead of pytest's default per-function `tmp_path`)
        keeps the number of directories created by the ~1200 SPy tests
        small, while still isolating files between test modules. The
        directory is removed after the module's tests finish, pass or fail.

    `av3c_image`
        Opens the sample image "92AV3C.lan". Found via the current
        directory or the `SPECTRAL_DATA` environment variable (see
        `spectral.io.spyfile.find_file_path`); not bundled with the repo.

    `gt`
        Opens the ground-truth classification image "92AV3GT.GIS" that
        accompanies "92AV3C.lan", as a plain ndarray. Found the same way.

Tests that only apply to an optional dependency (e.g., matplotlib- or
PIL-based graphics tests) should import it via `import_or_require` rather
than `pytest.importorskip` directly. By default a missing dependency skips
the test, same as `importorskip`. Passing "all", or a comma-separated list
of module names (e.g. "matplotlib,PIL"), to the --require-optional-deps
command line option (or the SPECTRAL_REQUIRE_OPTIONAL_DEPS environment
variable, if the option isn't given) turns a missing import for the named
module(s) into a hard test failure instead of a skip -- useful for a CI job
that's supposed to guarantee those tests actually ran. Run `pytest --help`
and look under the "spectral" group for the option's description.

The Qt/OpenGL 3D window tests (`test_hypercube.py`, `test_ndwindow.py`, the
"gui3d" marker) additionally need a Qt platform plugin that can actually
render a QOpenGLWidget -- Qt's "offscreen" plugin can construct a
QApplication with no display at all, but cannot host QOpenGLWidget, so a
real or virtual (Xvfb) display is needed for those specific tests. Use the
`require_gui3d` fixture to gate a test on that capability; it skips (or
fails, if 'PySide6' is in --require-optional-deps) when unavailable. Most
GUI3D interaction logic (construction, keyboard/mouse handling) doesn't
actually need real rendering and works fine under "offscreen".

All graphics/GUI tests -- both the 2D matplotlib/PIL-based tests
(`test_colorscale.py`, `test_graphics.py`, `test_spypylab.py`) and the 3D
"gui3d" tests above -- carry the "graphics" marker, so `-m "not graphics"`
excludes all of them regardless of which optional packages happen to be
installed.
'''

import importlib
import logging
import os
import shutil
import sys

import numpy as np
import pytest

import spectral as spy

try:
    # Force a non-interactive backend so spectral.graphics tests (imshow,
    # ImageView, ...) run headlessly without requiring a display.
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
AV3C_SPC = os.path.join(DATA_DIR, '92AV3C.spc')

# Populated by pytest_configure once the command line has been parsed. Kept
# as a module-level set (rather than threading `request`/`config` through
# every call site) since import_or_require is called at module import time
# by some test files, before any fixture would be available.
_required_optional_deps = set()


SKIPPABLE_OPTIONAL_DEPS = {
    'matplotlib': 'test_spypylab.py (2D matplotlib-based display)',
    'PIL': 'test_graphics.py (save_rgb/make_pil_image)',
    'PySide6': ("test_hypercube.py, test_ndwindow.py (construction/"
                "interaction); also governs whether those files' "
                "TestRendering classes skip or fail when no real OpenGL "
                "rendering surface is available"),
    'OpenGL': 'test_hypercube.py, test_ndwindow.py (construction/interaction)',
}


def pytest_addoption(parser):
    group = parser.getgroup('spectral')
    group.addoption(
        '--require-optional-deps',
        action='store',
        default=None,
        metavar='{%s,all}' % ','.join(SKIPPABLE_OPTIONAL_DEPS),
        help=(
            "Comma-separated names, or 'all', of optional dependencies for "
            "which spectral's tests should fail instead of skip if the "
            "dependency isn't installed (or, for 'PySide6', if no working "
            "OpenGL rendering surface is available). Skippable names: " +
            '; '.join('%s -- %s' % kv for kv in SKIPPABLE_OPTIONAL_DEPS.items()) +
            ". Defaults to the SPECTRAL_REQUIRE_OPTIONAL_DEPS environment "
            "variable if not given."
        ),
    )


def import_or_require(name):
    '''Imports `name`, skipping the caller if it isn't installed.

    See the module docstring for how --require-optional-deps /
    SPECTRAL_REQUIRE_OPTIONAL_DEPS changes a missing import from a skip
    into a failure.
    '''
    if 'all' in _required_optional_deps or name in _required_optional_deps:
        return importlib.import_module(name)
    return pytest.importorskip(name)


def pytest_configure(config):
    global _required_optional_deps
    value = config.getoption('--require-optional-deps')
    if value is None:
        value = os.environ.get('SPECTRAL_REQUIRE_OPTIONAL_DEPS', '')
    _required_optional_deps = {n.strip() for n in value.split(',') if n.strip()}

    # Several class-scoped fixtures (test_database.py, test_spyfile.py) are
    # defined as plain instance methods rather than `@classmethod`, because
    # `@pytest.fixture` on a `@classmethod` isn't supported by the older
    # pytest (8.x) that Python 3.8/3.9 CI resolves to. Revisit this filter
    # if a future pytest actually removes support for the instance-method
    # form (currently just a deprecation warning, still fully functional).
    config.addinivalue_line(
        'filterwarnings',
        'ignore:Class-scoped fixture defined as instance method is deprecated:DeprecationWarning',
    )
    config.addinivalue_line(
        'markers',
        'gui3d: Qt/OpenGL 3D window tests (hypercube.py, ndwindow.py).',
    )
    config.addinivalue_line(
        'markers',
        'graphics: all graphics/GUI tests, 2D and 3D (implies gui3d where '
        'applicable).',
    )


@pytest.fixture(autouse=True, scope='session')
def _quiet_spectral_logger():
    logging.getLogger('spectral').setLevel(logging.ERROR)


@pytest.fixture(autouse=True)
def _reset_matplotlib_state():
    yield
    # Only touch matplotlib if some test already imported pyplot (most of
    # the suite never does), to avoid paying import overhead on every test.
    if 'matplotlib.pyplot' in sys.modules:
        import matplotlib
        import matplotlib.pyplot as plt
        plt.close('all')
        matplotlib.rcdefaults()


@pytest.fixture(scope='module')
def testdir(tmp_path_factory, request):
    modname = request.module.__name__.rsplit('.', 1)[-1]
    path = tmp_path_factory.mktemp(modname)
    yield str(path)
    shutil.rmtree(str(path), ignore_errors=True)


@pytest.fixture
def av3c_image():
    return spy.open_image('92AV3C.lan')


@pytest.fixture
def gt():
    return np.array(spy.open_image('92AV3GT.GIS').read_band(0))


@pytest.fixture(scope='session')
def qapp():
    '''Session-scoped QApplication for the Qt/OpenGL 3D window tests.

    Constructing this only needs PySide6 to be importable and a Qt platform
    plugin to be available -- it works under Qt's "offscreen" platform with
    no real or virtual display. Actually rendering a QOpenGLWidget is a
    stronger requirement; see `gui3d_capable`/`require_gui3d`.
    '''
    import_or_require('PySide6')
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(scope='session')
def gui3d_capable(qapp):
    '''Whether this environment can actually render an OpenGL-backed Qt
    widget. A QApplication alone isn't sufficient -- e.g. Qt's "offscreen"
    platform plugin cannot host QOpenGLWidget -- so this probes for a real
    working GL surface by rendering a trivial widget and checking whether
    Qt actually invoked initializeGL on it.
    '''
    from PySide6.QtOpenGLWidgets import QOpenGLWidget

    class _Probe(QOpenGLWidget):
        rendered = False

        def initializeGL(self):
            self.rendered = True

    widget = _Probe()
    widget.resize(64, 64)
    widget.show()
    for _ in range(10):
        qapp.processEvents()
    ok = widget.rendered
    widget.close()
    return ok


@pytest.fixture
def require_gui3d(gui3d_capable):
    '''Skips (or fails, under `--require-optional-deps=PySide6`) a test
    that needs a real OpenGL-backed Qt widget to render.
    '''
    if gui3d_capable:
        return
    reason = ('No working OpenGL-backed Qt surface in this environment (no '
              'display/Xvfb, or the active Qt platform plugin does not '
              'support QOpenGLWidget). Run under a real display, or via '
              '`xvfb-run -a pytest ...`, to enable these tests.')
    if 'all' in _required_optional_deps or 'PySide6' in _required_optional_deps:
        pytest.fail(reason)
    pytest.skip(reason)
