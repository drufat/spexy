# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
"""
>>> from spexy.grid import Grid_1D
>>> from spexy.grid.grid import hodge_star_matrix
>>> g = Grid_1D.periodic(3)
>>> H0, H1, H0d, H1d = hodge_star_matrix(g)
>>> H0
array([[ 1.853,  0.121,  0.121],
       [ 0.121,  1.853,  0.121],
       [ 0.121,  0.121,  1.853]])
>>> H1d
array([[ 0.544, -0.033, -0.033],
       [-0.033,  0.544, -0.033],
       [-0.033, -0.033,  0.544]])
>>> H0d
array([[ 1.853,  0.121,  0.121],
       [ 0.121,  1.853,  0.121],
       [ 0.121,  0.121,  1.853]])
>>> H1
array([[ 0.544, -0.033, -0.033],
       [-0.033,  0.544, -0.033],
       [-0.033, -0.033,  0.544]])
>>> g = Grid_1D.periodic(8)
>>> H0, H1, H0d, H1d = hodge_star_matrix(g)
>>> H0
array([[ 0.683,  0.062, -0.016,  0.009, -0.008,  0.009, -0.016,  0.062],
       [ 0.062,  0.683,  0.062, -0.016,  0.009, -0.008,  0.009, -0.016],
       [-0.016,  0.062,  0.683,  0.062, -0.016,  0.009, -0.008,  0.009],
       [ 0.009, -0.016,  0.062,  0.683,  0.062, -0.016,  0.009, -0.008],
       [-0.008,  0.009, -0.016,  0.062,  0.683,  0.062, -0.016,  0.009],
       [ 0.009, -0.008,  0.009, -0.016,  0.062,  0.683,  0.062, -0.016],
       [-0.016,  0.009, -0.008,  0.009, -0.016,  0.062,  0.683,  0.062],
       [ 0.062, -0.016,  0.009, -0.008,  0.009, -0.016,  0.062,  0.683]])

Matrix is symmetric, circulant and centrosymmetric.
"""
import spexy.ops.num as num
import numpy as np
from spexy.ops.nat import (
    H, Hinv,
    S, Sinv,
    Q, Qinv,
)
from spexy.ops.num import (
    roll, slice_, weave,
)

xmin, xmax = 0, 2 * np.pi

even = lambda f: slice_(0, None, 2)(f)
odd = lambda f: slice_(1, None, 2)(f)
evenodd = lambda f: even(f) + odd(f)


def D0(f):
    return roll(-1)(f) - f


def D1(f):
    return 0


def D0d(f):
    return f - roll(+1)(f)


def D1d(f):
    return 0


def H0d(f):
    return H(f)


def H1(f):
    return Hinv(f)


def H0(f):
    return H(f)


def H1d(f):
    return Hinv(f)


Q, Qinv
S, Sinv


def Pup0(f):
    """
    >>> num.mat(Pup0, 2)
    array([[ 1. ,  0. ],
           [ 0.5,  0.5],
           [ 0. ,  1. ],
           [ 0.5,  0.5]])
    """
    # return np.real(sp.E_space(f.shape[0])(f))
    return weave(f, S(f))


def Pup0d(f):
    """
    >>> num.mat(Pup0d, 2)
    array([[ 0.5,  0.5],
           [ 1. ,  0. ],
           [ 0.5,  0.5],
           [ 0. ,  1. ]])
    """
    # return np.roll(T0(f), +1)
    return weave(Sinv(f), f)


def Pup1(f):
    return Pup0d(H1(f))


def Pup1d(f):
    return Pup0(H1d(f))


def Pdown0(f):
    return even(f)


def Pdown0d(f):
    return odd(f)


def Pdown1(f):
    return evenodd(Q(f))


def Pdown1d(f):
    """
    >>> Pdown1d([1, 1, 1, 1])
    array([ 3.142,  3.142])
    """
    return evenodd(roll(+1)(Q(f)))


def G0(f):
    """
    >>> G0([1, 1, 1])
    array([ 0.,  0.,  0.])
    """
    return num.G(f)


def G0d(f):
    """
    >>> G0d([1, 1, 1])
    array([ 0.,  0.,  0.])
    """
    return num.G(f)


derivative = (
    (D0, D1),
    (D0d, D1d),
)

hodge_star = (
    (H0, H1),
    (H0d, H1d),
)

upsample = (
    (Pup0, Pup1),
    (Pup0d, Pup1d),
)

downsample = (
    (Pdown0, Pdown1),
    (Pdown0d, Pdown1d),
)

gradient = (
    (G0, None),
    (G0d, None),
)


#######################
# Old Stuff
#######################

def wedge_explicit():
    """
    Return \alpha ^ \beta. Keep only for primal forms for now.
    """
    import spexy.spectral as sp

    def w00(alpha, beta):
        return alpha * beta

    def _w01(alpha, beta):
        """ This is the associative implementation. """
        return sp.S(sp.H(alpha * sp.Hinv(sp.Sinv(beta))))

    def w01(alpha, beta):
        """ This is the non-associative implementation. """
        # a = interweave(alpha, T(alpha, [S]))
        # b = interweave(T(beta, [Hinv, Sinv]), T(beta, [Hinv]))
        a = sp.refine(alpha)
        b = sp.refine(sp.Hinv(sp.Sinv(beta)))
        c = sp.S(sp.H(a * b))
        return c[0::2] + c[1::2]

    return w00, w01, _w01


def derivative_matrix(n):
    import spexy.helper
    rng = np.arange(n)
    ons = np.ones(n)
    d = np.row_stack((
        np.column_stack((
            rng,
            np.roll(rng, shift=-1),
            +ons)),
        np.column_stack((
            rng,
            rng,
            -ons))
    ))
    D = spexy.helper.sparse_matrix(d, n, n)
    return D, -D.T


def differentiation_toeplitz(n):
    """
    >>> differentiation_toeplitz(4)
    array([[ 1., -1.,  0.,  0.],
           [ 0.,  1., -1.,  0.],
           [ 0.,  0.,  1., -1.],
           [-1.,  0.,  0.,  1.]])
    """
    import scipy.linalg
    c = np.zeros(n)
    c[0] = +1
    c[-1] = -1
    D = scipy.linalg.circulant(c)
    return D


def hodge_star_toeplitz(g):
    """
    The Hodge-Star using a Toeplitz matrix.
    >>> from spexy.grid import Grid_1D
    >>> g = Grid_1D.periodic(3)
    >>> hodge_star_toeplitz(g)
    array([[ 1.853,  0.121,  0.121],
           [ 0.121,  1.853,  0.121],
           [ 0.121,  0.121,  1.853]])
    """
    import scipy.linalg
    import spexy.spectral as sp
    P0, P1, P0d, P1d = g.projection()
    column = P1d(lambda x: sp.alpha0(g.n, x))
    row = np.concatenate((column[:1], column[1:][::-1]))
    return scipy.linalg.toeplitz(column, row)
