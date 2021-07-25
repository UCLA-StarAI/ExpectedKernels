import numpy as np
from scipy.linalg import eigvals, norm, expm, svd, pinv
from scipy.spatial.distance import cdist
from copy import deepcopy

_EPS = 1e-6

def assert_eq(a, b, message=""):
    """Check if a and b are equal."""
    assert a == b, "Error: %s != %s ! %s" % (a, b, message)
    return


def assert_le(a, b, message=""):
    """Check if a and b are equal."""
    assert a <= b, "Error: %s > %s ! %s" % (a, b, message)
    return


def assert_ge(a, b, message=""):
    """Check if a and b are equal."""
    assert a >= b, "Error: %s < %s ! %s" % (a, b, message)
    return


def assert_len(l, length, message=""):
    """Check list/array l is of shape shape."""
    assert_eq(len(l), length)
    return


def assert_shape(A, shape, message=""):
    """Check array A is of shape shape."""
    assert_eq(A.shape, shape)
    return

def is_symmetric(A):
    """Check if A is a symmetric matrix."""
    assert isinstance(A, np.ndarray)
    return norm(A - A.T) < _EPS