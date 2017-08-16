# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import functools
import operator

import numpy as np


def sizes(shape):
    """
    >>> shape = ((3, 4), (5, 6), (2, 2))
    >>> sizes(shape)
    (12, 30, 4)
    """
    return tuple(functools.reduce(operator.mul, t) for t in shape)


def strides(shape):
    """
    >>> shape = (3, 4)
    >>> strides(shape)
    (4, 1)
    >>> shape = (3, 4, 2)
    >>> strides(shape)
    (8, 2, 1)
    """
    stride = functools.reduce(operator.mul, shape)

    def f(s):
        nonlocal stride
        stride = stride // s
        return stride

    return tuple(f(s) for s in shape)


def index_ij(shape):
    """
    >>> shape = (3, 4)
    >>> strides(shape)
    (4, 1)
    >>> idx, idxinv = index_ij(shape)
    >>> idx(0, 1)
    1
    >>> idxinv(1)
    (0, 1)
    >>> idx(1, 0)
    4
    >>> idxinv(4)
    (1, 0)
    >>> idx(2, 1)
    9
    >>> idxinv(9)
    (2, 1)
    >>> idx(1, 2)
    6
    >>> idxinv(6)
    (1, 2)
    >>> idx(np.array([1, 0, 1, 2]), np.array([0, 1, 2, 1]))
    array([4, 1, 6, 9])
    >>> idxinv(np.array([1, 3, 7, 5]))
    (array([0, 0, 1, 1]), array([1, 3, 3, 1]))
    >>> a = np.arange(12)
    >>> A = a.reshape(shape)
    >>> [A[idxinv(n)] for n in range(12)]
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    """
    strd = strides(shape)

    def idx(*indices):
        return sum(i * s for i, s in zip(indices, strd))

    def idxinv(n):
        i = n

        def f(s):
            nonlocal i
            _ = i // s
            i = i % s
            return _

        return tuple(f(s) for s in strd)

    return idx, idxinv


def index_k(sizes):
    """
    >>> sizes = (12, 30, 4)
    >>> tuple(np.cumsum(sizes))
    (12, 42, 46)
    >>> idx, idxinv = index_k(sizes)
    >>> idx(0), idx(11), idx(12), idx(41), idx(42), idx(43), idx(45)
    ((0, 0), (0, 11), (1, 0), (1, 29), (2, 0), (2, 1), (2, 3))
    >>> idxinv(0, 0), idxinv(0, 11), idxinv(1, 0), idxinv(1, 29), idxinv(2, 0), idxinv(2, 1), idxinv(2, 3)
    (0, 11, 12, 41, 42, 43, 45)
    >>> idx(-1)
    Traceback (most recent call last):
    ...
    KeyError
    >>> idx(46)
    Traceback (most recent call last):
    ...
    KeyError
    """
    bounds = np.cumsum((0,) + sizes)

    def idx(i):
        if i < bounds[0]:
            raise KeyError
        for k in range(len(bounds) - 1):
            if i < bounds[k + 1]:
                return k, i - bounds[k]
        raise KeyError

    def idxinv(k, i):
        return i + bounds[k]

    return idx, idxinv


def index_kij(shape):
    """
    >>> shapes = ((3, 4), (6, 5), (2, 2))
    >>> idx, idxinv = index_kij(shapes)
    >>> idx(0), idx(1), idx(11), idx(12), idx(41), idx(42)
    ((0, 0, 0), (0, 0, 1), (0, 2, 3), (1, 0, 0), (1, 5, 4), (2, 0, 0))
    >>> idxinv(0, 0, 0), idxinv(0, 0, 1), idxinv(0, 2, 3), idxinv(1, 0, 0), idxinv(1, 5, 4), idxinv(2, 0, 0)
    (0, 1, 11, 12, 41, 42)
    """
    idx_k, idxinv_k = index_k(sizes(shape))
    idx_ij, idxinv_ij = zip(*(index_ij(s) for s in shape))

    def idx(i):
        k, i = idx_k(i)
        ij = idxinv_ij[k](i)
        return (k,) + ij

    def idxinv(k, *ij):
        i = idx_ij[k](*ij)
        i = idxinv_k(k, i)
        return i

    return idx, idxinv


def countshape(shape):
    """
    >>> countshape((3, 4))
    12
    >>> countshape([(3, 4), (5, 6)])
    42
    >>> countshape([(12, 4), (5, 6), (3, 8)])
    102
    """
    if type(shape[0]) is tuple:
        return sum(functools.reduce(operator.mul, s) for s in shape)
    else:
        return countshape((shape,))


def forward_idx(c, n):
    idx, idxinv = index_kij(n)

    def _(m):
        k, i, j = idx(m)
        return c[k](i, j)

    return _


def intervals(shape):
    """
    >>> shape = ((4, 3), (7, 6), (2, 1))
    >>> intervals(shape)
    ((0, 12), (12, 54), (54, 56))
    """
    cumsum = np.cumsum(sizes(shape))
    a = (0,) + tuple(cumsum[:-1])
    b = tuple(cumsum)
    ab = tuple(zip(a, b))
    return ab


def forward(shape):
    """
    >>> shape = ((4, 3), (2, 1))
    >>> N = sum(sizes(shape))
    >>> F = forward(shape)
    >>> a = np.arange(N)
    >>> a
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13])
    >>> A = F(a)
    >>> A
    (array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11]]), array([[12],
           [13]]))
    >>> idx, idxinv = index_kij(shape)
    >>> [(lambda A, k, i, j: A[k][i, j])(A, *idx(n)) for n in range(N)]
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    >>> [a[idxinv(k, i, j)] for k, i, j in gen(shape)]
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    >>> [A[k][i, j] for k, i, j in gen(shape)]
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    >>> shape = ((4, 3),)
    >>> N = sum(sizes(shape))
    >>> F = forward(shape)
    >>> a = np.arange(N)
    >>> a
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
    >>> F(a)
    (array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11]]),)
    >>> F(a)[0][1, 2] = 100
    >>> a
    array([  0,   1,   2,   3,   4, 100,   6,   7,   8,   9,  10,  11])
    """
    a_b_s = tuple(zip(intervals(shape), shape))

    def F(f):
        return tuple(f[a:b].reshape(s) for (a, b), s in a_b_s)

    return F


def backward(shape):
    """
    >>> shape = ((4, 3), (2, 1))
    >>> N = sum(sizes(shape))
    >>> F = forward(shape)
    >>> Finv = backward(shape)
    >>> a = np.arange(N)
    >>> Finv(F(a))
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13])
    """
    abk = tuple(zip(intervals(shape), range(len(shape))))
    N = sum(sizes(shape))

    def Finv(f):
        g = np.empty(N, dtype=f[0].dtype)
        for (a, b), k in abk:
            g[a:b] = f[k].reshape(-1)
        return g

    return Finv


def index_k_vec(sizes):
    """
    >>> sizes = (12, 30, 4)
    >>> np.cumsum(sizes)
    array([12, 42, 46])
    >>> idx, idxinv = index_k_vec(sizes)
    >>> idx(np.array([ 0, 11, 12, 43, 45]))
    (array([0, 0, 1, 2, 2]), array([ 0, 11,  0,  1,  3]))
    >>> idxinv(np.array([ 0, 0, 1, 2, 2]), np.array([ 0, 11,  0,  1,  3]))
    array([ 0, 11, 12, 43, 45])
    >>> idx(np.array([-1]))
    Traceback (most recent call last):
    ...
    KeyError
    >>> idx(np.array([46]))
    Traceback (most recent call last):
    ...
    KeyError
    """
    bounds = np.cumsum((0,) + sizes)

    def idx(i):
        if np.any(i < bounds[0]) or np.any(i >= bounds[-1]):
            raise KeyError
        k = np.empty_like(i)
        for k_ in range(len(bounds) - 1):
            k[np.logical_and(bounds[k_] <= i, i < bounds[k_ + 1])] = k_
        return k, i - bounds[k]

    def idxinv(k, i):
        return i + bounds[k]

    return idx, idxinv


def index_kij_vec(shapes):
    """
    >>> shapes = ((3, 4), (6, 5), (2, 2))
    >>> idx, idxinv = index_kij_vec(shapes)
    >>> idx(np.array([0, 1, 11, 12, 41, 42]))
    (array([0, 0, 0, 1, 1, 2]), array([0, 0, 2, 0, 5, 0]), array([0, 1, 3, 0, 4, 0]))
    >>> idxinv(np.array([0, 0, 0, 1, 1, 2]), np.array([0, 0, 2, 0, 5, 0]), np.array([0, 1, 3, 0, 4, 0]))
    array([ 0,  1, 11, 12, 41, 42])
    """
    Nk = len(shapes)
    idx_k, idxinv_k = index_k_vec(sizes(shapes))
    idx_ij, idxinv_ij = zip(*(index_ij(shape) for shape in shapes))

    def idx(i):
        ij = tuple(np.empty_like(i) for _ in range(len(shapes[0])))
        k, i_ = idx_k(i)
        for k_ in range(Nk):
            for ij_, ijk_ in zip(ij, idxinv_ij[k_](i_[k == k_])):
                ij_[k == k_] = ijk_
        return (k,) + ij

    def idxinv(k, *ij):
        i = np.empty_like(k)
        for k_ in range(Nk):
            i[k == k_] = idx_ij[k_](*(_[k == k_] for _ in ij))
        return idxinv_k(k, i)

    return idx, idxinv


def gen(shape):
    return ((k, i, j) for k, s in enumerate(shape) for i in range(s[0]) for j in range(s[1]))


def test_gen():
    assert (
        tuple(gen(((1, 2), (3, 4))))
        ==
        (
            (0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3), (1, 1, 0),
            (1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 0), (1, 2, 1), (1, 2, 2), (1, 2, 3)
        )
    )
