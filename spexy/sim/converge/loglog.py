import numpy as np


def loglog(a, b, n):
    return np.exp(np.linspace(np.log(a), np.log(b), n))


def loglogint(a, b, n):
    """
    >>> loglogint(4, 40, 10)
    array([ 4,  5,  7,  9, 11, 14, 19, 24, 31, 40])
    """
    return np.unique(loglog(a, b, n).round(0).astype(np.int))
