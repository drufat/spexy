# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import operator
from functools import reduce
import numpy as np


def nested_tuple(atom=(lambda a: type(a) is not tuple)):
    """
    >>> index, unpack, pack = nested_tuple()
    >>> T = ((3, 4, 5), (6, (7, 8)))
    >>> index(T)
    ((0, 1, 2), (3, (4, 5)))
    >>> unpack(T)
    (3, 4, 5, 6, 7, 8)
    >>> T = (3, 4, 5, 6, 7, 8)
    >>> I = ((0, 1, 2), (3, (4, 5)))
    >>> pack(T, I)
    ((3, 4, 5), (6, (7, 8)))
    >>> T = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')
    >>> I = ((((0, 1, 2), 3, 4), 5, (6, 7)))
    >>> pack(T, I)
    ((('a', 'b', 'c'), 'd', 'e'), 'f', ('g', 'h'))
    >>> T = (1,)
    >>> I = 0
    >>> pack(T, I)
    1

    >>> atom = lambda a: type(a) == tuple and tuple(type(_) for _ in a) == (int, int)
    >>> index, unpack, pack = nested_tuple(atom)
    >>> T = (3, 4)
    >>> index(T)
    0
    >>> unpack(T)
    ((3, 4),)
    >>> T = ((3, 4), ((3, 4), (7, 8)))
    >>> index(T)
    (0, (1, 2))
    >>> unpack(T)
    ((3, 4), (3, 4), (7, 8))

    >>> atom = lambda a: type(a) == tuple and tuple(type(_) for _ in a) == (int, int)
    >>> a = (1, 2)
    >>> atom(a)
    True
    >>> index, unpack, pack = nested_tuple()
    >>> T = ((a, a, a), (a, a))
    >>> index(T)
    (((0, 1), (2, 3), (4, 5)), ((6, 7), (8, 9)))
    >>> unpack(T)
    (1, 2, 1, 2, 1, 2, 1, 2, 1, 2)
    >>> index, unpack, pack = nested_tuple(atom)
    >>> index(T)
    ((0, 1, 2), (3, 4))
    >>> unpack(((a, a, a), (a, a)))
    ((1, 2), (1, 2), (1, 2), (1, 2), (1, 2))

    """

    def index(T):
        i = -1

        def imp(T):
            nonlocal i
            if atom(T):
                i += 1
                return i
            if type(T) is tuple:
                return tuple(imp(t) for t in T)
            raise TypeError('Unrecognized atom {}'.format(T))

        return imp(T)

    def unpack(T):
        l = []

        def imp(T):
            if atom(T):
                l.append(T)
                return
            if type(T) is tuple:
                for t in T: imp(t)
                return
            raise TypeError('Unrecognized atom {}'.format(T))

        imp(T)
        return tuple(l)

    def pack(T, idx):

        def imp(I):
            if type(I) is int:
                return T[I]
            if type(I) is tuple:
                return tuple(imp(i) for i in I)
            raise TypeError('Unrecognized atom {}'.format(T))

        return imp(idx)

    return index, unpack, pack


def shape(X):
    """
    Given a structured array, return its shape.
    >>> e = lambda *args: np.empty(args)
    >>> shape(e(4, 5))
    (4, 5)
    >>> shape((e(4, 5),))
    ((4, 5),)
    >>> x = e(4, 5)
    >>> y = e(2, 3)
    >>> shape( (((x,), (y,)),) )
    ((((4, 5),), ((2, 3),)),)
    >>> shape((e(3, 2), e(4, 5)))
    ((3, 2), (4, 5))
    >>> shape((e(3, 2), e(4, 5), e(2, 3, 4)))
    ((3, 2), (4, 5), (2, 3, 4))
    >>> shape((e(1), (e(2), e(3))))
    ((1,), ((2,), (3,)))
    """
    if type(X) is np.ndarray:
        return X.shape

    if type(X) is tuple:
        return tuple(shape(x) for x in X)

    raise TypeError('Incorrect type {} for {}'.format(type(X), X))


def unshape(X, shape_=None):
    """
    Given a structured array, return a flat array and a shape tuple describing its structure.
    >>> a = np.array
    >>> x = a([[0, 1, 2],
    ...        [3, 4, 5]])
    >>> unshape(x, (2, 3))
    array([0, 1, 2, 3, 4, 5])
    >>> x = (a([[0, 1, 2],
    ...         [3, 4, 5]]),
    ...      a([[6, 7],
    ...         [8, 9]]))
    >>> unshape(x, ((2, 3), (2, 2)))
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> x = (a([[0, 1,],
    ...         [2, 3]]),
    ...      ( a([4, 5]),
    ...        a([6, 7]) ))
    >>> unshape(x)
    array([0, 1, 2, 3, 4, 5, 6, 7])
    """
    if shape_ and shape_ != shape(X):
        raise ValueError("Incorrect shape: {}!={}".format(shape(X), shape_))

    if type(X) is np.ndarray:
        return X.reshape(-1)

    if type(X) is tuple:
        return np.concatenate([unshape(x) for x in X])

    raise TypeError('Incorrect type {} for {}'.format(type(X), X))


def cumul_index(shape):
    """
    >>> cumul_index(((3,1), (4,5), (6,7)))
    (3, 23, 65)
    """
    size = np.array([reduce(operator.mul, s) for s in shape], dtype=int)
    return tuple(np.cumsum(size))


def arange2(shape):
    """
    >>> arange2((2,1))
    array([[0],
           [1]])
    >>> arange2((3,4))
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])
    >>> arange2(((2, 2), (3, 2)))
    (array([[0, 1],
           [2, 3]]), array([[4, 5],
           [6, 7],
           [8, 9]]))
    """
    if type(shape[0]) is int:
        n = cumul_index((shape,))[-1]
    else:
        n = cumul_index(shape)[-1]
    return reshape(np.arange(n, dtype=int), shape)


def index(i, shape):
    """
    >>> shape = ((11, 12), (13, 14))
    >>> A = arange2(shape)
    >>> k, (i, j) = index(0, shape)
    >>> A[k][i,j]
    0
    >>> k, (i, j) = index(1, shape)
    >>> A[k][i,j]
    1
    >>> k, (i, j) = index(10, shape)
    >>> A[k][i,j]
    10
    >>> k, (i, j) = index(200, shape)
    >>> A[k][i,j]
    200
    """
    cum = (0,) + cumul_index(shape)
    for (k, (a, b)) in enumerate(zip(cum[:-1], cum[1:])):
        if a <= i < b:
            return k, np.unravel_index(i - a, shape[k])
    raise ValueError


def shape_sizes(shape):
    """
    >>> shape_sizes((3, 1))
    3
    >>> shape_sizes(((3, 1), (4, 5), (6, 7)))
    (3, 20, 42)
    >>> shape_sizes(((1,2), ((3, 4), (5, 6))))
    (2, (12, 30))
    """

    def atom(a):
        return type(a) == tuple and tuple(type(_) for _ in a) == tuple(int for _ in a)

    index, unpack, pack = nested_tuple(atom)
    idx = index(shape)
    shape_ = unpack(shape)

    size_ = tuple(reduce(operator.mul, s) for s in shape_)

    return pack(size_, idx)


def shape_slices(shape):
    """
    >>> shape_slices((3, 1))
    (0, 3)
    >>> shape_slices(((3, 1), (4, 5), (6, 7)))
    ((0, 3), (3, 23), (23, 65))
    >>> shape_slices(((1,2), ((3, 4), (5, 6))))
    ((0, 2), ((2, 14), (14, 44)))
    """

    def atom(a):
        return type(a) == tuple and tuple(type(_) for _ in a) == (int, int)

    index, unpack, pack = nested_tuple(atom)
    idx = index(shape)
    shape_ = unpack(shape)

    size_ = tuple(reduce(operator.mul, s) for s in shape_)
    cum_ = tuple(np.cumsum((0,) + size_))
    slices_ = tuple(zip(cum_[:-1], cum_[1:]))

    return pack(slices_, idx)


def reshape(x, shape):
    """
    Given a flat array and a shape tuple, return a structured array.
    >>> reshape(np.array([0, 1, 2, 3, 4, 5]), (2, 3))
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> reshape(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), ((2, 3), (2, 2)))
    (array([[0, 1, 2],
           [3, 4, 5]]), array([[6, 7],
           [8, 9]]))
    >>> reshape(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), (((2, 3),), ((2, 2),)))
    ((array([[0, 1, 2],
           [3, 4, 5]]),), (array([[6, 7],
           [8, 9]]),))

    Make sure it is a view over the original array, and not a copy.
    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> shape = ((2, 3), (2, 2))
    >>> y = reshape(x, shape)
    >>> y[0][0, 0] = 1
    >>> y[1][0, 0] = 2
    >>> x
    array([1, 1, 2, 3, 4, 5, 2, 7, 8, 9])
    >>> y[0][:,:] += 1
    >>> x
    array([2, 2, 3, 4, 5, 6, 2, 7, 8, 9])
    >>> y[0][1,2] -= 2
    >>> x
    array([2, 2, 3, 4, 5, 4, 2, 7, 8, 9])
    """

    def atom(a):
        return type(a) == tuple and tuple(type(_) for _ in a) == tuple(int for _ in a)

    index, unpack, pack = nested_tuple(atom)
    idx = index(shape)
    shape_ = unpack(shape)

    size_ = tuple(reduce(operator.mul, s) for s in shape_)
    cum_ = tuple(np.cumsum((0,) + size_))
    slice_ = tuple(zip(cum_[:-1], cum_[1:]))

    x_ = tuple(x[slc[0]:slc[1]].reshape(shp) for slc, shp in zip(slice_, shape_))

    return pack(x_, idx)
