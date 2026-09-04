'''Tests of `spectral.utilities.convert_legacy_class_file`.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_convert_legacy_class_file.py
'''

import os
import pickle

import numpy as np
from numpy.testing import assert_allclose
import pytest

from spectral.algorithms.algorithms import TrainingClass, TrainingClassSet
from spectral.utilities.convert_legacy_class_file import (
    convert_legacy_class_file, main)


@pytest.fixture
def masked_classes_image():
    '''A synthetic 6x6x4 image with two labeled classes in `mask`.'''
    img = np.random.RandomState(0).rand(6, 6, 4)
    mask = np.zeros((6, 6), int)
    mask[0:2, 0:2] = 1   # 4 pixels
    mask[3:5, 3:5] = 2   # 4 pixels
    return (img, mask)


@pytest.fixture
def classes(masked_classes_image):
    (img, mask) = masked_classes_image
    classes = TrainingClassSet()
    classes.add_class(TrainingClass(img, mask, index=1))
    classes.add_class(TrainingClass(img, mask, index=2))
    classes.calc_stats()
    return classes


def write_legacy_class_file(fname, classes):
    '''Writes `classes` out using the old, pickle-based `save` format that
    `TrainingClassSet.save` used prior to switching to `.npz`.'''
    ids = sorted(classes.classes.keys())
    with open(fname, 'wb') as f:
        pickle.dump(classes.classes[ids[0]].mask, f)
        pickle.dump(len(classes), f)
        for id in ids:
            c = classes.classes[id]
            pickle.dump(c.index, f)
            pickle.dump(c.stats.cov, f)
            pickle.dump(c.stats.mean, f)
            pickle.dump(c.stats.nsamples, f)
            pickle.dump(c.class_prob, f)


class TestConvertLegacyClassFile:

    def test_convert_then_load_round_trip(self, classes, testdir,
                                          masked_classes_image):
        (img, _) = masked_classes_image
        legacy_fname = os.path.join(testdir, 'legacy.classes')
        new_fname = os.path.join(testdir, 'converted.classes')
        write_legacy_class_file(legacy_fname, classes)

        convert_legacy_class_file(legacy_fname, new_fname)

        loaded = TrainingClassSet()
        loaded.load(new_fname, img)
        assert len(loaded) == len(classes)
        for i in (1, 2):
            assert_allclose(loaded[i].stats.mean, classes[i].stats.mean)
            assert_allclose(loaded[i].stats.cov, classes[i].stats.cov)
            assert loaded[i].stats.nsamples == classes[i].stats.nsamples
            assert loaded[i].class_prob == classes[i].class_prob

    def test_rejects_malicious_pickle_payload(self, testdir, tmp_path):
        '''Conversion must not execute arbitrary code from a crafted
        legacy file, and must not produce an output file if it fails.'''
        marker = tmp_path / 'PWNED'

        class Exploit:
            def __reduce__(self):
                return (os.system, (f'touch {marker}',))

        legacy_fname = os.path.join(testdir, 'malicious.classes')
        new_fname = os.path.join(testdir, 'should_not_be_created.classes')
        with open(legacy_fname, 'wb') as f:
            pickle.dump(Exploit(), f)  # stands in for the `mask` field
            pickle.dump(0, f)          # nclasses

        with pytest.raises(pickle.UnpicklingError):
            convert_legacy_class_file(legacy_fname, new_fname)

        assert not marker.exists()
        assert not os.path.exists(new_fname)


class TestMain:
    '''Tests of the `python -m spectral.utilities.convert_legacy_class_file`
    command-line entry point.'''

    def test_main_converts_file_and_prints_confirmation(self, classes,
                                                         testdir,
                                                         masked_classes_image,
                                                         capsys):
        (img, _) = masked_classes_image
        legacy_fname = os.path.join(testdir, 'legacy_cli.classes')
        new_fname = os.path.join(testdir, 'converted_cli.classes')
        write_legacy_class_file(legacy_fname, classes)

        main([legacy_fname, new_fname])

        captured = capsys.readouterr()
        assert legacy_fname in captured.out
        assert new_fname in captured.out

        loaded = TrainingClassSet()
        loaded.load(new_fname, img)
        assert len(loaded) == len(classes)

    def test_main_requires_both_filenames(self):
        with pytest.raises(SystemExit):
            main(['only_one_arg.classes'])
