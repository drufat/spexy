# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np


def c(f):
    return np.ctypeslib.as_ctypes(f.ravel())


def dims(f):
    return {
        1: (lambda: (1, f.shape[0])),
        2: (lambda: f.shape),
    }[f.ndim]()


def wrap_op(H):
    """
    >>> from spexy.ops.nat import H, Hinv
    >>> f = np.ones(5)/np.pi
    >>> H(f)
    array([ 0.4,  0.4,  0.4,  0.4,  0.4])
    >>> H(f*2)
    array([ 0.8,  0.8,  0.8,  0.8,  0.8])
    >>> H(np.r_[[f], [f*2]])
    array([[ 0.4,  0.4,  0.4,  0.4,  0.4],
           [ 0.8,  0.8,  0.8,  0.8,  0.8]])
    >>> from spexy.ops.num import mat
    >>> mat(H, 3)
    array([[ 1.853,  0.121,  0.121],
           [ 0.121,  1.853,  0.121],
           [ 0.121,  0.121,  1.853]])
    >>> mat(Hinv, 3)
    array([[ 0.544, -0.033, -0.033],
           [-0.033,  0.544, -0.033],
           [-0.033, -0.033,  0.544]])
    """

    def _(f):
        f = np.ascontiguousarray(f, dtype='double')
        M, N = dims(f)
        out = np.empty_like(f)
        H(M, N, c(f), c(out))
        return out

    return _


def wrap_diff(imp):
    """
    >>> from spexy.ops.num import mat
    >>> from spexy.ops.nat import diff
    >>> mat(diff(), 4)
    array([[-1.,  1.,  0.,  0.],
           [ 0., -1.,  1.,  0.],
           [ 0.,  0., -1.,  1.]])
    >>> f = np.arange(9, dtype='double').reshape(3, 3)
    >>> diff()(f)
    array([[ 1.,  1.],
           [ 1.,  1.],
           [ 1.,  1.]])
    >>> diff()(f.T).T
    array([[ 3.,  3.,  3.],
           [ 3.,  3.,  3.]])
    """

    def diff():
        def _(f):
            f = np.ascontiguousarray(f, dtype='double')
            M, N = dims(f)
            Nout = N - 1
            shape = list(f.shape)
            shape[-1] = Nout
            out = np.empty(shape, dtype='double')
            imp(
                M,
                N, c(f),
                Nout, c(out),
            )
            return out

        return _

    return diff


def wrap_roll(imp):
    """
    >>> from spexy.ops.num import mat
    >>> from spexy.ops.nat import roll
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
    >>> f = np.arange(9, dtype='double').reshape(3, 3)
    >>> roll(1)(f)
    array([[ 2.,  0.,  1.],
           [ 5.,  3.,  4.],
           [ 8.,  6.,  7.]])
    >>> roll(1)(f.T).T
    array([[ 6.,  7.,  8.],
           [ 0.,  1.,  2.],
           [ 3.,  4.,  5.]])
    """

    def roll(n):
        def _(f):
            f = np.ascontiguousarray(f, dtype='double')
            M, N = dims(f)
            out = np.empty_like(f)
            imp(
                n,
                M,
                N, c(f),
                N, c(out),
            )
            return out

        return _

    return roll


def clamp(v, N):
    """
    >>> clamp(-100, 10)
    0
    >>> clamp(-10, 10)
    0
    >>> clamp(-9, 10)
    1
    >>> clamp(-1, 10)
    9
    >>> clamp(0, 10)
    0
    >>> clamp(1, 10)
    1
    >>> clamp(9, 10)
    9
    >>> clamp(10, 10)
    10
    >>> clamp(100, 10)
    10
    """
    if v < 0:
        return max(0, v + N)
    else:
        return min(v, N)


def outsize(N, begin, end, step):
    """
    >>> answer = lambda tup: len(list(range(tup[0]))[slice(*tup[1:])])
    >>> tup = (10, 1, -1, 1)
    >>> assert outsize(*tup)[0] == answer(tup)
    >>> tup = (2, 1, -1, 1)
    >>> assert outsize(*tup)[0] == answer(tup)
    >>> tup = (1, 1, -1, 1)
    >>> assert outsize(*tup)[0] == answer(tup)
    >>> tup = (10, 0, 10, 1)
    >>> assert outsize(*tup)[0] == answer(tup)
    >>> tup = (10, None, None, -1)
    >>> assert outsize(*tup)[0] == answer(tup)
    >>> tup = (10, -1, None, -1)
    >>> assert outsize(*tup)[0] == answer(tup)
    >>> tup = (10, -1, -5, -1)
    >>> assert outsize(*tup)[0] == answer(tup)
    """
    if step is None:
        step = 1

    if step > 0:
        if begin is None:
            begin = 0
        else:
            begin = clamp(begin, N)
        if end is None:
            end = N
        else:
            end = clamp(end, N)
        if begin > end: begin = end
        Nout = (end - begin - 1) // step + 1
        return Nout, begin, end, step

    if step < 0:
        if begin is None:
            begin = N - 1
        else:
            begin = clamp(begin, N)
        if end is None:
            end = -1
        else:
            end = clamp(end, N)
        Nout = (begin - end) // abs(step)
        return Nout, begin, end, step


def wrap_slice_(imp):
    """
    >>> from spexy.ops.num import mat
    >>> from spexy.ops.nat import slice_
    >>> mat(slice_(0, None, 2), 6)
    array([[ 1.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  0.]])
    >>> mat(slice_(1, None, 2), 6)
    array([[ 0.,  1.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  1.]])
    >>> mat(slice_(1, -1, None), 6)
    array([[ 0.,  1.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  0.]])
    >>> mat(slice_(None, None, -1), 6)
    array([[ 0.,  0.,  0.,  0.,  0.,  1.],
           [ 0.,  0.,  0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.,  0.]])
    """

    def slice_(begin_, end_, step_):
        def _(f):
            f = np.ascontiguousarray(f, dtype='double')
            M, N = dims(f)
            Nout, begin, _, step = outsize(N, begin_, end_, step_)
            shape = list(f.shape)
            shape[-1] = Nout
            out = np.empty(shape, dtype='double')
            imp(
                begin, step,
                M,
                N, c(f),
                Nout, c(out),
            )
            return out

        return _

    return slice_


def wrap_weave(imp):
    """ Interweave two arrays.
    >>> from spexy.ops.nat import weave
    >>> weave([0, 1, 2], [3, 4, 5])
    array([ 0.,  3.,  1.,  4.,  2.,  5.])
    >>> weave([0, 1, 2], [3, 4])
    array([ 0.,  3.,  1.,  4.,  2.])
    >>> weave([[0,1],[2,3]],[[4,5],[6,7]])
    array([[ 0.,  4.,  1.,  5.],
           [ 2.,  6.,  3.,  7.]])
    """

    def weave(in0, in1):
        in0 = np.ascontiguousarray(in0, dtype='double')
        in1 = np.ascontiguousarray(in1, dtype='double')
        M, Nin0 = dims(in0)
        _, Nin1 = dims(in1)
        assert (M == _)
        Nout = Nin0 + Nin1
        shape = list(in0.shape)
        shape[-1] = Nout
        out = np.empty(shape, dtype='double')
        imp(
            M,
            Nin0, c(in0),
            Nin1, c(in1),
            Nout, c(out)
        )
        return out

    return weave


def wrap_concat(imp):
    """ Interweave two arrays.
    >>> from spexy.ops.nat import concat
    >>> concat([0, 1, 2], [3, 4, 5])
    array([ 0.,  1.,  2.,  3.,  4.,  5.])
    >>> concat([[0,1],[2,3]],[[4],[5]])
    array([[ 0.,  1.,  4.],
           [ 2.,  3.,  5.]])
   """

    def concat(in0, in1):
        in0 = np.ascontiguousarray(in0, dtype='double')
        in1 = np.ascontiguousarray(in1, dtype='double')
        M, Nin0 = dims(in0)
        _, Nin1 = dims(in1)
        assert (M == _)
        Nout = Nin0 + Nin1
        shape = list(in0.shape)
        shape[-1] = Nout
        out = np.empty(shape, dtype='double')
        imp(
            M,
            Nin0, c(in0),
            Nin1, c(in1),
            Nout, c(out)
        )
        return out

    return concat
