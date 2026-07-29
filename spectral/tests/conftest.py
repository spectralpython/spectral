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
'''

import logging
import os
import shutil

import numpy as np
import pytest

import spectral as spy

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
AV3C_SPC = os.path.join(DATA_DIR, '92AV3C.spc')


def pytest_configure(config):
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
