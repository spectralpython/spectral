'''Converts legacy (pre-`.npz`) `TrainingClassSet` files to the current format.

Versions of SPy older than the one that introduced this module saved
:class:`~spectral.algorithms.algorithms.TrainingClassSet` data with
`pickle`. Unpickling a file can execute arbitrary code embedded in it if
the file has been tampered with or comes from an untrusted source, so
`TrainingClassSet.load` no longer reads that legacy format directly.

**Only call `convert_legacy_class_file` on files whose origin you trust.**
It still has to unpickle the input file to read it. It does so through
`_RestrictedLegacyTrainingClassUnpickler`, which restricts
`pickle.Unpickler.find_class` to a small whitelist of `numpy` array/scalar
reconstruction functions -- enough to represent the mask/index/covariance/
mean/sample-count/class-probability data a `TrainingClassSet` file
contains, and nothing that can construct an arbitrary object or invoke
arbitrary code. This is defense in depth, not a safety guarantee: it
blocks known pickle exploitation techniques, not necessarily future ones.
If you are not confident a file is one your own code produced (or one
that came from someone you trust), do not run it through this module --
regenerate the training data from its source instead.

This module is also runnable as a command-line script:

    python -m spectral.utilities.convert_legacy_class_file \\
        legacy_file.classes converted_file.classes
'''
from __future__ import annotations

import argparse
import pickle
from typing import Any

from ..algorithms.algorithms import GaussianStats, TrainingClass, TrainingClassSet

# Globals that a legacy (pre-npz) TrainingClassSet file is allowed to
# reference. Restricting `Unpickler.find_class` to this whitelist prevents
# `convert_legacy_class_file` from being used to execute arbitrary code via
# a crafted file, since only numpy array/scalar reconstruction is permitted.
_ALLOWED_LEGACY_TRAINING_CLASS_PICKLE_GLOBALS = frozenset({
    ('numpy', 'ndarray'),
    ('numpy', 'dtype'),
    ('numpy.core.multiarray', '_reconstruct'),
    ('numpy.core.multiarray', 'scalar'),
    ('numpy.core.numeric', '_frombuffer'),
    ('numpy._core.multiarray', '_reconstruct'),
    ('numpy._core.multiarray', 'scalar'),
    ('numpy._core.numeric', '_frombuffer'),
})


class _RestrictedLegacyTrainingClassUnpickler(pickle.Unpickler):
    '''Unpickler for legacy `TrainingClassSet` files that only allows
    constructing numpy arrays/scalars, so loading an untrusted file cannot
    directly construct an arbitrary object or execute arbitrary code. This
    is defense in depth, not a guarantee of safety -- see the module
    docstring.'''
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _ALLOWED_LEGACY_TRAINING_CLASS_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(
                f'Refusing to unpickle disallowed global: {module}.{name}')
        return super().find_class(module, name)


def convert_legacy_class_file(legacy_filename: str, new_filename: str) -> None:
    '''Converts a `TrainingClassSet` file saved by an old (pre-`.npz`)
    version of SPy into the current file format.

    Arguments:

        `legacy_filename` (str):

            Path to a file written by the old, `pickle`-based
            `TrainingClassSet.save`.

        `new_filename` (str):

            Path to write the converted file to, in the `numpy` `.npz`
            format now used by `TrainingClassSet.save`/
            `TrainingClassSet.load`.

    SECURITY WARNING: only call this function on files whose origin you
    trust. Reading `legacy_filename` requires unpickling it and, although
    a restricted unpickler is used that only permits constructing numpy
    arrays/scalars (blocking construction of arbitrary objects or
    execution of arbitrary code via the usual pickle exploitation
    techniques), unpickling an untrusted or tampered file is inherently
    risky. If you are not confident of the file's origin, do not convert
    it -- regenerate the training class data from its source instead.

    After conversion, load `new_filename` with
    `TrainingClassSet.load` as usual.
    '''
    with open(legacy_filename, 'rb') as f:
        unpickler = _RestrictedLegacyTrainingClassUnpickler(f)
        mask = unpickler.load()
        nclasses = unpickler.load()
        classes = TrainingClassSet()
        for _ in range(nclasses):
            index = unpickler.load()
            cov = unpickler.load()
            mean = unpickler.load()
            nsamples = unpickler.load()
            class_prob = unpickler.load()
            c = TrainingClass(None, mask, index, class_prob)
            c.stats = GaussianStats(mean=mean, cov=cov, nsamples=nsamples)
            if not (cov is None or mean is None or nsamples is None):
                c.stats_valid(True)
                c.nbands = len(mean)
            classes.add_class(c)
    classes.save(new_filename)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog='python -m spectral.utilities.convert_legacy_class_file',
        description=(
            'Converts a spectral.algorithms.TrainingClassSet file saved by '
            "an old version of SPy into the format read by today's "
            'TrainingClassSet.load(). Old versions of SPy serialized this '
            "data with Python's pickle module (such a file is commonly, "
            "though not necessarily, given a '.pkl' extension); current "
            "versions instead use a numpy '.npz' archive (commonly given "
            "a '.npz' extension, though TrainingClassSet.save()/.load() "
            "accept any filename you choose). TrainingClassSet.load() no "
            "longer reads the old pickle-based format directly, so a "
            "legacy file must be converted with this script before it can "
            'be loaded again.'),
        epilog=(
            'SECURITY WARNING: only run this on a legacy file whose '
            'origin you trust. Converting it still requires unpickling '
            'it; this script does so through a restricted unpickler that '
            'only permits constructing numpy arrays/scalars, blocking '
            'construction of arbitrary objects or execution of arbitrary '
            'code via the usual pickle exploitation techniques -- but '
            'that is defense in depth, not a guarantee of safety. If you '
            "are not confident of a file's origin, do not convert it; "
            'regenerate the training class data from its source instead.'))
    parser.add_argument(
        'legacy_filename',
        help="Path to the legacy, pickle-based file to convert (often, "
             "but not necessarily, named with a '.pkl' extension).")
    parser.add_argument(
        'new_filename',
        help="Path to write the converted file to, in the current "
             "numpy '.npz'-based format (often, but not necessarily, "
             "named with a '.npz' extension).")
    args = parser.parse_args(argv)
    convert_legacy_class_file(args.legacy_filename, args.new_filename)
    print(f"Converted '{args.legacy_filename}' to '{args.new_filename}'.")


if __name__ == '__main__':
    main()
