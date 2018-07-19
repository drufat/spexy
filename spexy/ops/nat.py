# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np
import spexy.ops.nat_raw as nr


def c(f):
    return np.ascontiguousarray(f, dtype='double')


def dims(f):
    return {
        1: (lambda: (1, f.shape[0])),
        2: (lambda: f.shape),
    }[f.ndim]()


freq = nr.freq


def wrap_op(func):
    """
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
        f = c(f)
        M, N = dims(f)
        out = np.empty_like(f)
        func(M, N, f, out)
        return out

    return _


[
    H, Hinv,
    S, Sinv,
    Q, Qinv,
    G,
] = [wrap_op(F) for F in [
    nr.H, nr.Hinv,
    nr.S, nr.Sinv,
    nr.Q, nr.Qinv,
    nr.G,
]]


def diff():
    """
    >>> from spexy.ops.num import mat
    >>> f = np.arange(9, dtype='double').reshape(3, 3)
    >>> diff()(f)
    array([[ 1.,  1.],
           [ 1.,  1.],
           [ 1.,  1.]])
    >>> diff()(f.T).T
    array([[ 3.,  3.,  3.],
           [ 3.,  3.,  3.]])
    >>> mat(diff(), 4)
    array([[-1.,  1.,  0.,  0.],
           [ 0., -1.,  1.,  0.],
           [ 0.,  0., -1.,  1.]])
    """

    def _(f):
        f = c(f)
        M, N = dims(f)
        Nout = N - 1
        shape = list(f.shape)
        shape[-1] = Nout
        out = np.empty(shape, dtype=f.dtype)
        nr.diff(
            M,
            N, f,
            Nout, out,
        )
        return out

    return _


def roll(n):
    """
    >>> from spexy.ops.num import mat
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

    def _(f):
        f = c(f)
        M, N = dims(f)
        out = np.empty_like(f)
        nr.roll(
            n,
            M,
            N, f,
            N, out,
        )
        return out

    return _


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


def slice_(begin_, end_, step_):
    """
    >>> from spexy.ops.num import mat
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

    def _(f):
        f = c(f)
        M, N = dims(f)
        Nout, begin, _, step = outsize(N, begin_, end_, step_)
        shape = list(f.shape)
        shape[-1] = Nout
        out = np.empty(shape, dtype=f.dtype)
        nr.slice_(
            begin, step,
            M,
            N, f,
            Nout, out,
        )
        return out

    return _


def weave(in0, in1):
    """ Interweave two arrays.
    >>> weave([0, 1, 2], [3, 4, 5])
    array([ 0.,  3.,  1.,  4.,  2.,  5.])
    >>> weave([0, 1, 2], [3, 4])
    array([ 0.,  3.,  1.,  4.,  2.])
    >>> weave([[0,1],[2,3]],[[4,5],[6,7]])
    array([[ 0.,  4.,  1.,  5.],
           [ 2.,  6.,  3.,  7.]])
    """

    in0 = c(in0)
    in1 = c(in1)
    M, Nin0 = dims(in0)
    _, Nin1 = dims(in1)
    assert (M == _)
    Nout = Nin0 + Nin1
    shape = list(in0.shape)
    shape[-1] = Nout
    out = np.empty(shape, dtype=in0.dtype)
    nr.weave(
        M,
        Nin0, in0,
        Nin1, in1,
        Nout, out,
    )
    return out


def concat(in0, in1):
    """ Concatenate two arrays.
    >>> concat([0, 1, 2], [3, 4, 5])
    array([ 0.,  1.,  2.,  3.,  4.,  5.])
    >>> concat([[0,1],[2,3]],[[4],[5]])
    array([[ 0.,  1.,  4.],
           [ 2.,  3.,  5.]])
   """

    in0 = c(in0)
    in1 = c(in1)
    M, Nin0 = dims(in0)
    _, Nin1 = dims(in1)
    assert (M == _)
    Nout = Nin0 + Nin1
    shape = list(in0.shape)
    shape[-1] = Nout
    out = np.empty(shape, dtype=in0.dtype)
    nr.concat(
        M,
        Nin0, in0,
        Nin1, in1,
        Nout, out,
    )
    return out
