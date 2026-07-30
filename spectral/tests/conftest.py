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


def pytest_addoption(parser):
    group = parser.getgroup('spectral')
    group.addoption(
        '--require-optional-deps',
        action='store',
        default=None,
        metavar='DEPS',
        help=(
            "Comma-separated optional dependency module names (e.g. "
            "'matplotlib,PIL'), or 'all', for which spectral's graphics "
            "tests should fail instead of skip if the dependency isn't "
            "installed. Defaults to the SPECTRAL_REQUIRE_OPTIONAL_DEPS "
            "environment variable if not given."
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
