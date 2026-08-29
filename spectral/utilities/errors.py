
import numpy as np


class SpyException(Exception):
    '''Base class for spectral module-specific exceptions.'''
    pass


class NaNValueWarning(UserWarning):
    pass


class NaNValueError(ValueError):
    pass


def has_nan(X):
    '''returns True if ndarray `X` contains a NaN value.'''
    return bool(np.isnan(np.min(X)))
