# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np
from functools import wraps
import spexy.bases.cardinals.num as cn


################
# Batch Mode
################

def batch(H):
    """
    >>> A = lambda a: np.concatenate(([a[0]], a[1:-1:2] + a[2:-1:2], [a[-1]]))
    >>> batch(A)(np.eye(6)).T
    array([[ 1.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  1.]])
    """

    @wraps(H)
    def _(f):
        return np.apply_along_axis(H, -1, f)

    return _


def batchall(f):
    def _():
        return tuple(batch(H) for H in f())

    return _


def mat(op, n):
    return op(np.eye(n)).T


################
# Helpers
################

def diff():
    """
    >>> mat(diff(), 4).round(3)
    array([[-1.,  1.,  0.,  0.],
           [ 0., -1.,  1.,  0.],
           [ 0.,  0., -1.,  1.]])
    """

    def _(f):
        return np.diff(f, axis=-1)

    return _


def roll(n):
    """
    >>> mat(roll(1), 4)
    array([[ 0.,  0.,  0.,  1.],
           [ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.]])
    >>> mat(roll(-1), 4)
    array([[ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.],
           [ 1.,  0.,  0.,  0.]])
    """

    def _(f):
        return np.roll(f, n, axis=-1)

    return _


def slice_(start, stop, step):
    """
    >>> mat(slice_(0, None, 2), 6)
    array([[ 1.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  0.]])
    >>> mat(slice_(1, None, 2), 6)
    array([[ 0.,  1.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  1.]])
    >>> mat(slice_(1, 4, 2), 6)
    array([[ 0.,  1.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.,  0.]])
    """

    def _(f):
        if len(f.shape) == 1:
            return f[start:stop:step]
        if len(f.shape) == 2:
            return f[:, start:stop:step]

    return _


def weave(a, b):
    """ Interweave two arrays.
    >>> weave([0, 1, 2], [3, 4, 5])
    array([0, 3, 1, 4, 2, 5])
    >>> weave([0, 1, 2], [3, 4])
    array([0, 3, 1, 4, 2])
    >>> weave([[0,1],[2,3]],[[4,5],[6,7]])
    array([[0, 4, 1, 5],
           [2, 6, 3, 7]])
    >>> weave([[0,1],[2,3]],[[4],[5]])
    array([[0, 4, 1],
           [2, 5, 3]])
    """
    a = np.asarray(a)
    b = np.asarray(b)

    shape = list(a.shape)
    shape[-1] = a.shape[-1] + b.shape[-1]
    out = np.empty(shape, dtype=a.dtype)

    if len(shape) == 1:
        out[0::2] = a
        out[1::2] = b
        return out

    if len(shape) == 2:
        out[:, 0::2] = a
        out[:, 1::2] = b
        return out


def concat(a, b):
    """
    >>> concat([0, 1, 2], [3, 4, 5])
    array([0, 1, 2, 3, 4, 5])
    >>> concat([[0,1],[2,3]],[[4],[5]])
    array([[0, 1, 4],
           [2, 3, 5]])
   """
    return np.hstack([a, b])


rev = slice_(None, None, -1)
mid = slice_(1, -1, None)


def mirror(typ=0, sign=+1):
    """
    >>> mirror(0,+1)(np.array([1, 2, 3]))
    array([1, 2, 3, 2])
    >>> mirror(0,-1)(np.array([1, 2, 3]))
    array([ 1,  2,  3, -2])
    >>> mirror(1,+1)(np.array([1, 2, 3]))
    array([1, 2, 3, 3, 2, 1])
    >>> mirror(1,-1)(np.array([1, 2, 3]))
    array([ 1,  2,  3, -3, -2, -1])

    """
    if typ == 0:
        return lambda f: concat(f, sign * mid(rev(f)))
    if typ == 1:
        return lambda f: concat(f, sign * rev(f))


def unmirror(typ=0):
    """
    >>> unmirror(0)(np.array([1, 2, 3, 2]))
    array([1, 2, 3])
    >>> unmirror(1)(np.array([1, 2, 3, -3, -2, -1]))
    array([1, 2, 3])
    """
    if typ == 0:
        return lambda f: slice_(None, (f.shape[-1] // 2 + 1), None)(f)
    if typ == 1:
        return lambda f: slice_(None, (f.shape[-1] // 2), None)(f)


def A00(f):
    """
    >>> A00([1, 2, 3])
    array([0, 1, 2, 3, 0])
    >>> A00([[1, 2], [3, 4]])
    array([[0, 1, 2, 0],
           [0, 3, 4, 0]])
    """
    f = np.asarray(f)

    if len(f.shape) == 1:
        return np.hstack([[0], f, [0]])

    if len(f.shape) == 2:
        [M, _] = f.shape
        z = np.zeros([M, 1], dtype=f.dtype)
        return np.hstack([z, f, z])


def B(C, x):
    """
    >>> a = np.array
    >>> B(cn.CU, -1)(a([1, 1, 1]))
    array([ 1.])
    >>> B(cn.CU, +1)(a([1, 1, 1]))
    array([ 1.])
    >>> s = np.sqrt(2)/2
    >>> B(cn.CU, -1)(a([-s, 0, s]))
    array([-1.])

    >>> B(cn.CT, -1)(a([1, 1, 1]))
    array([ 1.])
    >>> B(cn.CT, -1)(a([1, 1, 1]))
    array([ 1.])
    >>> s = np.sqrt(3)/2
    >>> B(cn.CT, -1)(a([-s, 0, s]))
    array([-1.])
    """

    def _(f):
        N = f.shape[-1]
        n = np.arange(N).reshape(-1, 1)
        return np.dot(
            f,
            C(N, n, x),
        )

    return _


BL = B(cn.CU, -1)
BR = B(cn.CU, +1)


def Abb(f):
    """
    >>> Abb(np.array([1, 2, 3]))
    array([ 0.586,  1.   ,  2.   ,  3.   ,  3.414])
    >>> Abb(np.array([[1, 1, 1], [1, 1, 1]]))
    array([[ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.]])
    >>> mat(Abb, 3)
    array([[ 1.707, -1.   ,  0.293],
           [ 1.   ,  0.   ,  0.   ],
           [ 0.   ,  1.   ,  0.   ],
           [ 0.   ,  0.   ,  1.   ],
           [ 0.293, -1.   ,  1.707]])
    """
    return np.hstack([BL(f), f, BR(f)])


##############################
# Converting to Fourier Space
##############################


def fourier(Op):
    @wraps(Op)
    def _(f):
        f = np.fft.fft(f)
        f = Op(f)
        f = np.fft.ifft(f)
        f = np.real(f)
        return f

    return _


################
# Fourier Ops
################

def freq(N):
    """
    >>> freq(1)
    array([0])
    >>> freq(2)
    array([ 0, -1])
    >>> freq(3)
    array([ 0,  1, -1])
    >>> freq(4)
    array([ 0,  1, -2, -1])
    >>> freq(5)
    array([ 0,  1,  2, -2, -1])
    >>> np.fft.fftshift(freq(5))
    array([-2, -1,  0,  1,  2])
    """
    i = np.arange(N)
    return np.select([
        i < (N + 1) // 2,
        True
    ], [
        i,
        i - N
    ])


def H_hat(N):
    k = freq(N)
    return np.select(
        [
            k == 0,
            True
        ], [
            2 * np.pi / N,
            np.sin(np.pi * k / N) * 2 / k
        ])


def Q_hat(N):
    k = freq(N)
    return np.select(
        [
            k == 0,
            True
        ], [
            2 * np.pi / N,
            (np.exp(2j * k * np.pi / N) - 1) / (1j * k)
        ])


def S_hat(N, s=+1):
    k = freq(N)
    return np.exp(s * 1j * k * np.pi / N)


def G_hat(N):
    k = freq(N)
    return 1j * k


################
# H
################


@batch
@fourier
def H(f):
    r"""

    .. math::
        \mathbf{H}^0 = \tilde{\mathbf{H}}^0 =
        \mathcal{F}^{-1} \mathbf{I}^{-\frac{h}{2}, \frac{h}{2}} \mathcal{F}

    >>> mat(H, 1)
    array([[ 6.283]])
    >>> mat(H, 2)
    array([[ 2.571,  0.571],
           [ 0.571,  2.571]])
    >>> mat(H, 3)
    array([[ 1.853,  0.121,  0.121],
           [ 0.121,  1.853,  0.121],
           [ 0.121,  0.121,  1.853]])
    """
    N = f.shape[0]
    return f * H_hat(N)


@batch
@fourier
def Hinv(f):
    r"""

    .. math::
        \mathbf{H}^1 = \tilde{\mathbf{H}}^1 =
        \mathcal{F}^{-1}{\mathbf{I}^{-\frac{h}{2},\frac{h}{2}}}^{-1}\mathcal{F}

    >>> mat(Hinv, 1)
    array([[ 0.159]])
    >>> mat(Hinv, 2)
    array([[ 0.409, -0.091],
           [-0.091,  0.409]])
    >>> mat(Hinv, 3)
    array([[ 0.544, -0.033, -0.033],
           [-0.033,  0.544, -0.033],
           [-0.033, -0.033,  0.544]])
    """
    N = f.shape[0]
    return f / H_hat(N)


################
# Q
################

@batch
@fourier
def Q(f):
    """
    >>> mat(Q, 2)
    array([[ 1.571,  1.571],
           [ 1.571,  1.571]])
    """
    N = f.shape[0]
    return f * Q_hat(N)


@batch
@fourier
def Qinv(f):
    """
    >>> mat(Qinv, 2)
    array([[ 0.159,  0.159],
           [ 0.159,  0.159]])
    """
    N = f.shape[0]
    return f / Q_hat(N)


################
# S
################

@batch
@fourier
def S(f):
    """
    >>> mat(S, 2)
    array([[ 0.5,  0.5],
           [ 0.5,  0.5]])
    >>> mat(S, 3)
    array([[ 0.667,  0.667, -0.333],
           [-0.333,  0.667,  0.667],
           [ 0.667, -0.333,  0.667]])
    """
    N = f.shape[0]
    return f * S_hat(N, +1)


@batch
@fourier
def Sinv(f):
    """
    >>> mat(Sinv, 2)
    array([[ 0.5,  0.5],
           [ 0.5,  0.5]])
    >>> mat(Sinv, 3)
    array([[ 0.667, -0.333,  0.667],
           [ 0.667,  0.667, -0.333],
           [-0.333,  0.667,  0.667]])
    """
    N = f.shape[0]
    return f * S_hat(N, -1)


################
# G
################

@batch
@fourier
def G(f):
    """
    >>> mat(G, 2)
    array([[ 0.,  0.],
           [ 0.,  0.]])
    >>> mat(G, 3)
    array([[ 0.   ,  0.577, -0.577],
           [-0.577,  0.   ,  0.577],
           [ 0.577, -0.577,  0.   ]])
    >>> mat(G, 4)
    array([[ 0. ,  0.5,  0. , -0.5],
           [-0.5,  0. ,  0.5,  0. ],
           [ 0. , -0.5,  0. ,  0.5],
           [ 0.5,  0. , -0.5,  0. ]])
    """
    N = f.shape[0]
    return f * G_hat(N)


####################
#  Weight
####################
def W(f):
    """
    >>> W(np.array([1, 1, 1]))
    array([ 0.,  1.,  0.])
    >>> W(np.array([[1, 1, 1],[1, 1, 1]]))
    array([[ 0.,  1.,  0.],
           [ 0.,  1.,  0.]])
    >>> mat(W, 3)
    array([[ 0.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  0.]])
    """
    N = f.shape[-1]
    i = np.arange(N)
    s = np.sin(i * np.pi / (N - 1))
    return f * s


def Winv(f):
    """
    >>> Winv(np.array([1, 1, 1]))[1:-1]
    array([ 1.])
    """
    N = f.shape[-1]
    i = np.arange(N)
    s = np.sin(i * np.pi / (N - 1))
    return f / s


def Wd(f):
    """
    >>> Wd(np.array([1, 1]))
    array([ 0.707,  0.707])
    >>> mat(Wd, 3)
    array([[ 0.5,  0. ,  0. ],
           [ 0. ,  1. ,  0. ],
           [ 0. ,  0. ,  0.5]])
    """
    N = f.shape[-1]
    i = np.arange(N) + 0.5
    s = np.sin(i * np.pi / N)
    return f * s


def Wdinv(f):
    """
    >>> Wdinv(np.array([1, 1]))
    array([ 1.414,  1.414])
    >>> mat(Wdinv, 3)
    array([[ 2.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  2.]])
    """
    N = f.shape[-1]
    i = np.arange(N) + 0.5
    s = np.sin(i * np.pi / N)
    return f / s
