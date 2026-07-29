'''Shared pytest fixtures for the SPy test suite.

Common fixtures:

    `testdir`
        A per-module temp directory for files written during tests. Using
        module scope (instead of pytest's default per-function `tmp_path`)
        keeps the number of directories created by the ~1200 SPy tests
        small, while still isolating files between test modules. The
        directory is removed after the module's tests finish, pass or fail.

    `av3c_image`
        Opens the sample image "92AV3C.lan" (bundled under `tests/data/`)
        used by most of the suite.

    `gt`
        Opens the ground-truth classification image "92AV3GT.GIS" that
        accompanies "92AV3C.lan", as a plain ndarray.
'''

import logging
import os
import shutil

import numpy as np
import pytest

import spectral as spy

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
AV3C_LAN = os.path.join(DATA_DIR, '92AV3C.lan')
AV3GT_GIS = os.path.join(DATA_DIR, '92AV3GT.GIS')
AV3C_SPC = os.path.join(DATA_DIR, '92AV3C.spc')


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
    return spy.open_image(AV3C_LAN)


@pytest.fixture
def gt():
    return np.array(spy.open_image(AV3GT_GIS).read_band(0))
