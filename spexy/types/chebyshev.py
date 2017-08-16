# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
"""
>>> from spexy.grid import Grid_1D
>>> from spexy.grid.grid import hodge_star_matrix
>>> g = Grid_1D.chebold(1)
>>> H0, H1, H0d, H1d = hodge_star_matrix(g)
>>> H0
array([[ 0.75,  0.25],
       [ 0.25,  0.75]])
>>> H1d
array([[ 1.5, -0.5],
       [-0.5,  1.5]])
>>> g = Grid_1D.chebold(2)
>>> H0, H1, H0d, H1d = hodge_star_matrix(g)
>>> H0d
array([[ 0.854,  0.146],
       [ 0.146,  0.854]])
>>> H1
array([[ 1.207, -0.207],
       [-0.207,  1.207]])
"""
import spexy.spectral as sp
import numpy as np
from spexy.ops import num, nat

imp = nat

xmin, xmax = -1, +1


@sp.batch
def D0(f):
    return np.diff(f)


def D1(f):
    return 0


@sp.batch
def D0d(f):
    return np.diff(np.concatenate(([0], f, [0])))


def D1d(f):
    return 0


@sp.batch
def H0d(f):
    """
    >>> sp.to_matrix(H0d, 2)
    array([[ 0.854,  0.146],
           [ 0.146,  0.854]])
    """
    f = f * sp.Sin_d(f.shape[0])
    f = sp.mirror1(f, -1)
    f = sp.H(f)
    f = sp.unmirror1(f)
    return f


@sp.batch
def H1(f):
    """
    >>> sp.to_matrix(H1, 2)
    array([[ 1.207, -0.207],
           [-0.207,  1.207]])
    """
    f = sp.mirror1(f, -1)
    f = sp.Hinv(f)
    f = sp.unmirror1(f)
    f = f / sp.Sin_d(f.shape[0])
    return f


@sp.batch
def H0(f):
    """
    >>> sp.to_matrix(H0, 2)
    array([[ 0.75,  0.25],
           [ 0.25,  0.75]])
    """
    f = sp.mirror0(f, +1)
    f = sp.F(f)
    f = sp.fourier_K(f, 0, 0.5)
    f = sp.Finv(f)
    f = np.real(f)
    f = sp.fold0(f, -1)
    return f


@sp.batch
def H1d(f):
    """
    This op requires 10 Fourier Transforms, when the others
    require only two. Is there no simler way to implement it?

    >>> sp.to_matrix(H1d, 2)
    array([[ 1.5, -0.5],
           [-0.5,  1.5]])
    >>> sp.to_matrix(H1d, 3)
    array([[ 4.5  , -0.328,  0.5  ],
           [-0.5  ,  0.914, -0.5  ],
           [ 0.5  , -0.328,  4.5  ]])
    """
    N = f.shape[0]
    b = sp.half_edge_base(N)
    fl, fr = f[[0, -1]]

    f = np.concatenate([[0], f[1:-1], [0]])
    f = sp.mirror0(f, -1)
    f = sp.Hinv(f)

    def midpoints(f):
        f = sp.S(f)
        f = f / sp.Omega_d(f.shape[0])
        f = sp.Sinv(f)
        f = sp.unmirror0(f)
        return f

    def endpoints(f):
        ll = fl - sp.I_space(0, .5)(f)[0]
        rr = fr - sp.I_space(-.5, 0)(f)[N - 1]
        return ll * b + rr * b[::-1]

    f = midpoints(f) + endpoints(f)
    return f


@sp.batch
def S(f):
    """
    Interpolate from primal to dual vertices.
    >>> sp.to_matrix(S, 2)
    array([[ 0.5,  0.5]])
    >>> sp.to_matrix(S, 3)
    array([[ 0.604,  0.5  , -0.104],
           [-0.104,  0.5  ,  0.604]])
    """
    f = sp.mirror0(f, +1)
    f = imp.S(f)
    f = sp.unmirror1(f)
    return f


@sp.batch
def Sinv(f):
    """
    Interpolate from dual to primal vertices.
    Since there are smaller number of dual vertices,
    this is only a pseudo inverse of S_cheb, and not an
    exact inverse.
    >>> np.allclose(
    ...     Sinv(np.array([ -1/np.sqrt(2),  1/np.sqrt(2)])),
    ...     np.array([ -1, 0, 1]))
    True
    >>> np.allclose(
    ...     Sinv(np.array([ 1/np.sqrt(2),  -1/np.sqrt(2)])),
    ...     np.array([ 1, 0, -1]))
    True
    >>> sp.to_matrix(Sinv, 1)
    array([[ 1.],
           [ 1.]])
    >>> sp.to_matrix(Sinv, 2)
    array([[ 1.207, -0.207],
           [ 0.5  ,  0.5  ],
           [-0.207,  1.207]])
    """
    f = sp.mirror1(f, +1)
    f = imp.Sinv(f)
    f = sp.unmirror0(f)
    return f


@sp.batch
def Q(f):
    """
    Interpolate from primal vertices to primal edges.
    >>> sp.to_matrix(Q, 2)
    array([[ 1.,  1.]])
    >>> sp.to_matrix(Q, 3)
    array([[ 0.417,  0.667, -0.083],
           [-0.083,  0.667,  0.417]])
    """
    f = sp.mirror0(f, +1)
    f = sp.F(f)
    f = sp.fourier_K(f, 0.0, 1.0)
    f = sp.Finv(f)
    f = np.real(f)
    f = sp.unmirror1(f)
    return f


@sp.batch
def Qinv(f):
    """
    Interpolate from primal edges to primal vertices.
    >>> sp.to_matrix(Qinv, 2)
    array([[ 1.5, -0.5],
           [ 0.5,  0.5],
           [-0.5,  1.5]])
    >>> sp.to_matrix(Qinv, 3)
    array([[ 3.167, -0.833,  0.5  ],
           [ 1.   ,  0.667, -0.333],
           [-0.333,  0.667,  1.   ],
           [ 0.5  , -0.833,  3.167]])
    """
    f = sp.mirror1(f, -1)
    f = sp.F(f)
    f = sp.fourier_K_inv(f, 0.0, 1.0)
    f = sp.Finv(f)
    f = sp.unmirror0(f)
    return f


@sp.batch
def Pup0(f):
    return imp.weave(f, S(f))


@sp.batch
def Pup0d(f):
    return imp.weave(Sinv(f), f)


@sp.batch
def Pup1(f):
    return Pup0d(H1(f))


@sp.batch
def Pup1d(f):
    return Pup0(H1d(f))


@sp.batch
def Pdown0(f):
    return f[0::2]


@sp.batch
def Pdown0d(f):
    return f[1::2]


@sp.batch
def Pdown1(f):
    f = Q(f)
    return f[0::2] + f[1::2]


@sp.batch
def Pdown1d(f):
    f = Q(f)
    f = np.concatenate([[0], f, [0]])
    return f[0::2] + f[1::2]


@sp.batch
def G0(f):
    """
    >>> num.mat(G0, 2)
    array([[-0.5,  0.5],
           [-0.5,  0.5]])
    >>> num.mat(G0, 3)
    array([[-1.5,  2. , -0.5],
           [-0.5,  0. ,  0.5],
           [ 0.5, -2. ,  1.5]])
    """
    N = f.shape[0]
    f = num.mirror(0, +1)(f)
    u = np.fft.fft(f)
    f = num.G(f)
    f = num.unmirror(0)(f)
    f = num.Winv(f)

    prime = np.ones(N)
    prime[0] = .5
    prime[-1] = .5
    n = np.arange(N)
    f[0] = - sum(prime * n ** 2 * u[:N]) / (N - 1)
    f[-1] = - sum(prime * (-np.ones(N)) ** (n + 1) * n ** 2 * u[:N]) / (N - 1)
    return f


def G0d(f):
    """
    >>> num.mat(G0d, 2)
    array([[-0.707,  0.707],
           [-0.707,  0.707]])
    """
    f = num.mirror(1, +1)(f)
    f = imp.G(f)
    f = num.unmirror(1)(f)
    f = num.Wdinv(f)
    return f


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
