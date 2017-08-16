# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import itertools
import math
from collections import namedtuple
from importlib import util as iu

import numpy as np
from scipy.sparse import csr_matrix
from spexy.ops.num import weave


def logistic(k=1, x0=0):
    return lambda x: 1 / (1 + np.exp(-k * (x - x0)))


def nCr(n, r):
    """
    >>> nCr(1, 0)
    1
    >>> nCr(1, 1)
    1
    >>> nCr(2, 0)
    1
    >>> nCr(2, 1)
    2
    >>> nCr(2, 2)
    1
    >>> nCr(3, 0)
    1
    >>> nCr(3, 1)
    3
    >>> nCr(3, 2)
    3
    >>> nCr(3, 3)
    1
    """
    f = math.factorial
    return f(n) // f(r) // f(n - r)


def approx(a, b):
    c = np.round_(a, 10) == np.round_(b, 10)
    if type(c) == np.ndarray:
        return c.all()
    return c


def to_matrix(f, N):
    """
    >>> np.array_equal(
    ... to_matrix(lambda x: x, 2),
    ... [[ 1.,  0.],
    ...  [ 0.,  1.]])
    True
    >>> np.array_equal(
    ... to_matrix(lambda x: np.roll(x,+1), 2),
    ... [[ 0.,  1.],
    ...  [ 1.,  0.]])
    True
    """
    M = np.vstack(f(b) for b in np.eye(N)).T
    if approx(np.real(M), M).all():
        M = np.real(M)
    return M


def to_matrix_by_index(f, N):
    """
    >>> np.array_equal(
    ... to_matrix_by_index(lambda i, j: 10*i + j, 2),
    ... [[  0.,   1.],
    ...  [ 10.,  11.]])
    True
    """
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M[i, j] = f(i, j)
    return M


def sparse_diag(x):
    """
    >>> np.array_equal(
    ... sparse_diag([1,2,3]).todense(),
    ... np.matrix([[1, 0, 0],
    ...            [0, 2, 0],
    ...            [0, 0, 3]]))
    True
    """
    x = np.asanyarray(x)
    N = len(x)
    ii = np.row_stack([np.arange(N)] * 2)
    return csr_matrix((x, ii), shape=(N, N))


def sparse_matrix(ijk, N, M):
    """
    >>> np.array_equal(
    ... sparse_matrix([[0,1,1], [1,0,2]], 2, 2).todense(),
    ... np.matrix([[0, 1],
    ...            [2, 0]]))
    True
    """
    ijk = np.asanyarray(ijk)
    ij = ijk[:, :2].T
    k = ijk[:, 2].T
    return csr_matrix((k, ij), shape=(N, M))


class bunch(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def nametuple(name, fields, dictionary):
    """
    >>> nametuple('Point', 'x,y', dict(x=1,y=2,z=3))
    Point(x=1, y=2)
    """
    d = {x: dictionary[x] for x in fields.split(',')}
    return namedtuple(name, fields)(**d)


def NamedTuple(attribs):
    """
    >>> def Point():
    ...    This = NamedTuple("x,y")
    ...    x = 1
    ...    y = 2
    ...    return This()
    >>> Point()
    Point(x=1, y=2)
    """
    import inspect
    outerframe = inspect.currentframe().f_back
    name = outerframe.f_code.co_name
    NamedTuple = namedtuple(name, attribs)

    def This():
        local = inspect.currentframe().f_back.f_locals

        def fields():
            for field in NamedTuple._fields:
                yield local.get(field)

        return NamedTuple._make(fields())

    return This


def set_attributes(obj, namedtuple):
    for f in namedtuple.__class__._fields:
        setattr(obj, f, getattr(namedtuple, f))


def set_dictionary(obj, namedtuple):
    for f in namedtuple.__class__._fields:
        obj[f] = getattr(namedtuple, f)


def this_function_name():
    """
    >>> def f():
    ...    return this_function_name()
    >>> f()
    'f'
    >>> a = f
    >>> a()
    'f'
    """
    import inspect
    outerframe = inspect.currentframe().f_back
    name = outerframe.f_code.co_name
    return name


def make_namedtuple(metaclass, dictionary):
    """
    Create a named tuple from a dictionary which is a superset of the fields

    >>> Point = namedtuple('Point', 'x,y')
    >>> make_namedtuple(Point, {'x':1, 'y':2, 'z':3})
    Point(x=1, y=2)
    """
    return metaclass._make((dictionary[f] for f in metaclass._fields))


def apply_map(faces, identify):
    """
    >>> apply_map([0,1,2], {0:3, 1:4, 2:5})
    array([3, 4, 5])
    """
    faces = np.asanyarray(faces)
    data = faces.ravel()
    mp = np.arange(0, max(data) + 1)
    mp[list(identify.keys())] = list(identify.values())
    data = mp[data]
    return data.reshape(faces.shape)


def bndry_with_zeros(a, axis=-1):
    """
    Tab `a` with zeros at both ends along `axis`.
    >>> x = [1, 2, 4, 7, 0]
    >>> bndry_with_zeros(x)
    array([0, 1, 2, 4, 7, 0, 0])
    >>> x = [[1, 3, 6, 10], [0, 5, 6, 8]]
    >>> bndry_with_zeros(x, axis=0)
    array([[ 0,  0,  0,  0],
           [ 1,  3,  6, 10],
           [ 0,  5,  6,  8],
           [ 0,  0,  0,  0]])
    >>> bndry_with_zeros(x, axis=-1)
    array([[ 0,  1,  3,  6, 10,  0],
           [ 0,  0,  5,  6,  8,  0]])
    """
    a = np.asanyarray(a)
    f1d = lambda x: np.concatenate(([0], x, [0]))
    return np.apply_along_axis(f1d, axis, a)


def is_equidistant(x):
    """
    >>> is_equidistant((0,1,2))
    True
    >>> is_equidistant((0,1,2.5))
    False
    """
    d = np.diff(x)
    return (np.round_(d, 8) == np.round_(d[0], 8)).all()


#################################
# Multiple dispatch
#################################

def multipledispatch(T):
    def apply_decorator(dispatch_fn):
        __multi__ = {}

        def _inner(*args):
            d = tuple(f.degree for f in args)
            c = tuple(f.components for f in args)
            d_ = dispatch_fn(*d)
            c_ = __multi__[d](*c)
            return T(d_, c_)

        _inner.__multi__ = __multi__
        _inner.__default__ = None
        return _inner

    return apply_decorator


def register(dispatch_fn, *dispatch_key):
    def apply_decorator(fn):
        if not dispatch_key:
            dispatch_fn.__default__ = fn
        else:
            dispatch_fn.__multi__[dispatch_key] = fn

    return apply_decorator


def subdivide(x):
    """
    Subdivide x like below. Assume points are equidistant.
    *   *   *   *   *   *
    * * * * * * * * * * *
    """
    if len(x) < 2: return x
    assert is_equidistant(x)
    y = weave(x[:-1], x[:-1] + 0.5 * np.diff(x))
    return np.concatenate((y, [x[-1]]))


def eq(*args):
    """
    >>> eq(1, 1)
    True
    >>> eq(1, 2)
    False
    >>> eq(1, 2, 3)
    False
    >>> eq(1, 1, 1)
    True

    """
    for l, r in zip(args[:-1], args[1:]):
        if not np.allclose(l, r):
            return False
    return True


def func_eq(*funcs):
    """
    >>> import numpy as np
    >>> def f(x): return np.sqrt(1 - x**2)
    >>> def g(x): return np.sin(np.arccos(x))
    >>> def h(x): return np.sqrt(1 - x) * np.sqrt(1 + x)
    >>> x = np.linspace(-1, +1, 10)
    >>> func_eq(f, g, h)(x)
    True
    """

    def _(*args):
        return eq(*tuple(func(*args) for func in funcs))

    return _


def assignfunctionto(d):
    def assign(k):
        def _(f):
            d[k] = f

        return _

    return assign


def apply_operators(H, axis):
    def get_apply(h):
        return lambda x: np.apply_along_axis(h, axis, x)

    if type(H) is tuple or type(H) is list:
        return tuple(get_apply(h) for h in H)
    elif type(H) is dict:
        return {k: get_apply(H[k]) for k in H}
    else:
        raise TypeError


def cartesian_product(X, Y):
    """
    >>> cartesian_product([0, 1],[2, 3])
    (array([0, 1, 0, 1]), array([2, 2, 3, 3]))
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    X, Y = [x.flatten() for x in np.meshgrid(X, Y)]
    return X, Y


def split(t):
    """
    >>> split([1, 2, 3, 4])
    ([1, 2], [3, 4])
    """
    N = len(t)
    return t[:N // 2], t[N // 2:]


def peek(gen):
    """
    >>> gen = iter(range(3))
    >>> gen, first = peek(gen)
    >>> first
    0
    >>> list(gen)
    [0, 1, 2]
    """
    first = next(gen)
    gen = itertools.chain([first], gen)
    return gen, first


def load_source(name, src):
    """
    Load source from `src` into a module `name`.
    """
    spec = iu.spec_from_file_location(name, src)
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m