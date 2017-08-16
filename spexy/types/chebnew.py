# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
"""
>>> from spexy.grid import Grid_1D
>>> from spexy.grid.grid import hodge_star_matrix
>>> g = Grid_1D.chebnew(3)
>>> H0, H1, H0d, H1d = hodge_star_matrix(g)
>>> H0
array([[ 0.808,  0.058],
       [ 0.058,  0.808]])
>>> H1d
array([[ 1.244, -0.089],
       [-0.089,  1.244]])
>>> H0d
array([[ 0.411,  0.111, -0.022],
       [ 0.056,  0.889,  0.056],
       [-0.022,  0.111,  0.411]])
>>> H1
array([[ 2.488, -0.333,  0.179],
       [-0.167,  1.167, -0.167],
       [ 0.179, -0.333,  2.488]])

>>> g = Grid_1D.chebnew(8)
>>> H0, H1, H0d, H1d = hodge_star_matrix(g)
>>> H0
array([[ 0.134,  0.019, -0.005,  0.002, -0.001,  0.   , -0.   ],
       [ 0.01 ,  0.243,  0.026, -0.006,  0.002, -0.001,  0.   ],
       [-0.002,  0.02 ,  0.317,  0.029, -0.005,  0.002, -0.   ],
       [ 0.001, -0.004,  0.027,  0.343,  0.027, -0.004,  0.001],
       [-0.   ,  0.002, -0.005,  0.029,  0.317,  0.02 , -0.002],
       [ 0.   , -0.001,  0.002, -0.006,  0.026,  0.243,  0.01 ],
       [-0.   ,  0.   , -0.001,  0.002, -0.005,  0.019,  0.134]])

The matrix is a Centrosymmetric matrix
"""
from spexy.ops import nat, num

xmin, xmax = -1, +1

imp = nat
diff = imp.diff()
even = imp.slice_(0, None, 2)
odd = imp.slice_(1, None, 2)
evenodd = lambda f: even(f) + odd(f)
mid = imp.slice_(1, -1, None)


def D0(f):
    return diff(num.A00(f))


def D1(f):
    return 0


def D0d(f):
    return diff(f)


def D1d(f):
    return 0


def H0d(f):
    """
    >>> num.mat(H0d, 3)
    array([[ 0.411,  0.111, -0.022],
           [ 0.056,  0.889,  0.056],
           [-0.022,  0.111,  0.411]])
    """
    f = num.Wd(f)
    f = num.mirror(1, -1)(f)
    f = imp.H(f)
    f = num.unmirror(1)(f)
    return f


def H1(f):
    """
    >>> num.mat(H1, 3)
    array([[ 2.488, -0.333,  0.179],
           [-0.167,  1.167, -0.167],
           [ 0.179, -0.333,  2.488]])
    """
    f = num.mirror(1, -1)(f)
    f = imp.Hinv(f)
    f = num.unmirror(1)(f)
    f = num.Wdinv(f)
    return f


def H0(f):
    """
    >>> num.mat(H0, 2)
    array([[ 0.808,  0.058],
           [ 0.058,  0.808]])
    """
    f = num.A00(f)
    f = num.W(f)

    f = num.mirror(0, -1)(f)
    f = imp.H(f)
    f = num.unmirror(0)(f)

    f = mid(f)
    return f


def H1d(f):
    """
    >>> num.mat(H1d, 2)
    array([[ 1.244, -0.089],
           [-0.089,  1.244]])
    """
    f = num.A00(f)

    f = num.mirror(0, -1)(f)
    f = imp.Hinv(f)
    f = num.unmirror(0)(f)

    f = num.Winv(f)
    f = mid(f)

    return f


def S(f):
    """
    >>> num.mat(S, 1)
    array([[ 1.],
           [ 1.]])
    >>> num.mat(S, 2)
    array([[ 1.366, -0.366],
           [ 0.5  ,  0.5  ],
           [-0.366,  1.366]])
    """
    f = num.Abb(f)
    f = num.mirror(0, +1)(f)
    f = imp.S(f)
    f = num.unmirror(1)(f)
    return f


def Sinv(f):
    """
    Interpolate from primal to dual vertices.
    >>> from spexy.grid import Grid_1D
    >>> from spexy.grid.grid import switch_matrix
    >>> switch_matrix(Grid_1D.chebnew(2))[0]
    array([[ 1.],
           [ 1.]])
    >>> num.mat(Sinv, 2)
    array([[ 0.5,  0.5]])
    >>> switch_matrix(Grid_1D.chebnew(3))[0]
    array([[ 1.366, -0.366],
           [ 0.5  ,  0.5  ],
           [-0.366,  1.366]])
    >>> num.mat(Sinv, 3)
    array([[ 0.455,  0.667, -0.122],
           [-0.122,  0.667,  0.455]])
    """
    f = num.mirror(1, +1)(f)
    f = imp.Sinv(f)
    f = num.unmirror(0)(f)
    f = mid(f)
    return f


def Pup0(f):
    return imp.weave(S(f), f)


def Pup0d(f):
    return imp.weave(f, Sinv(f))


def Pup1(f):
    return Pup0d(H1(f))


def Pup1d(f):
    return Pup0(H1d(f))


def Q(f):
    """
    >>> num.mat(Q, 1)
    array([[ 1.],
           [ 1.]])
    >>> num.mat(Q, 2)
    array([[ 0.625, -0.125],
           [ 0.5  ,  0.5  ],
           [-0.125,  0.625]])
    """
    # f = addbndry(f)
    # f = chebyshev.Q(f)
    # return f

    f = num.A00(f)
    f = num.W(f)

    f = num.mirror(0, -1)(f)
    f = imp.Q(f)
    f = num.unmirror(1)(f)

    return f


def Qinv(f):
    """
    >>> num.mat(Qinv, 2)
    array([[ 0.5,  0.5]])
    >>> num.mat(Qinv, 3)
    array([[ 1.   ,  0.667, -0.333],
           [-0.333,  0.667,  1.   ]])
    """
    # f = chebyshev.Qinv(f)
    # f = f[1:-1]
    # return f

    f = num.mirror(1, -1)(f)
    f = imp.Qinv(f)
    f = num.unmirror(0)(f)

    f = num.Winv(f)
    f = mid(f)

    return f


def Pdown0(f):
    return odd(f)


def Pdown0d(f):
    return even(f)


def Pdown1(f):
    return evenodd(Q(f))


def Pdown1d(f):
    return evenodd(mid(Q(f)))


def G0(f):
    """
    >>> num.mat(G0, 3)
    array([[-2.121,  2.828, -0.707],
           [-0.707, -0.   ,  0.707],
           [ 0.707, -2.828,  2.121]])
    """
    f = num.Abb(f)
    f = num.mirror(0, +1)(f)
    f = num.G(f)
    f = num.unmirror(0)(f)
    f = num.Winv(f)
    f = mid(f)
    return f


def G0d(f):
    """
    >>> num.mat(G0d, 2)
    array([[-0.707,  0.707],
           [-0.707,  0.707]])
    """
    f = num.mirror(1, +1)(f)
    f = num.G(f)
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
