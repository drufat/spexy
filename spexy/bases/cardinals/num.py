# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import spexy  # to set print options

import numpy as np
from numpy import (sin, cos, pi)

spexy

S = lambda _: _
half = 0.5
acos = np.arccos
Abs = np.abs


def Eq(x, y): return np.isclose(x, y)


def Piecewise(*args): return np.select(*tuple(zip(*args))[::-1])


###########################
# Chebyshev Polynomials
###########################

def T(n, x):
    """
    >>> T(5, np.linspace(-1, 1, 4))
    array([-1.   , -0.992,  0.992,  1.   ])
    """
    theta = acos(x)
    return cos(n * theta)


def U(n, x):
    """
    >>> U(5, np.linspace(-1, 1, 4))
    array([-6.   , -0.947,  0.947,  6.   ])
    """
    theta = acos(x)
    return Piecewise(
        (n + 1, x >= +1.0),
        ((n + 1) * (-1.0) ** n, x <= -1.0),
        (sin((n + 1) * theta) / sin(theta), True)
    )


def Uclamp(n, x):
    """
    >>> Uclamp(5, np.linspace(-1, 1, 4))
    array([ 0.   ,  0.922, -0.922,  0.   ])
    """
    return (T(n - 2, x) - T(n, x)) / S(2)


def Tclamp(n, x):
    """
    >>> Tclamp(5, np.linspace(-1, 1, 4))
    array([ 0.   ,  0.757, -0.757,  0.   ])
    """
    return T(n - 2, x) / S(2) - (T(Abs(n - 4), x) + T(n, x)) / S(4)


###########################
# First Derivatives
###########################

def dT(n, x):
    """
    >>> def _(n): return dT(n, xT(n, np.arange(n)))
    >>> _(2)
    array([-2.828,  2.828])
    >>> _(3)
    array([ 6., -3.,  6.])
    >>> _(4)
    array([-10.453,   4.33 ,  -4.33 ,  10.453])
    """
    return n * U(n - 1, x)


def dU(n, x):
    """
    >>> def _(n): return dU(n, xU(n, np.arange(n)))
    >>> _(2)
    array([-4.,  4.])
    >>> _(3)
    array([ 8., -4.,  8.])
    >>> _(4)
    array([-14.472,   5.528,  -5.528,  14.472])
    """

    def b(n): return n * (n + 1) * (n + 2) / 3

    return Piecewise(
        (b(n) * (-1) ** (n + 1), Eq(x, -1.0)),
        (b(n), Eq(x, +1.0)),
        (((n + 1) * T(n + 1, x) - x * U(n, x)) / (x ** 2 - 1), True)
    )


def dUclamp(n, x):
    """
    >>> def _(n): return dUclamp(n, xUclamp(n, np.arange(n)))
    >>> _(2)
    array([ 2., -2.])
    >>> _(3)
    array([-4.,  2., -4.])
    """
    return (dT(n - 2, x) - dT(n, x)) / 2


def dTclamp(n, x):
    """
    >>> def _(n): return dTclamp(n, xTclamp(n, np.arange(n)))
    >>> _(2)
    array([ 2., -2.])
    >>> _(3)
    array([-2.,  1., -2.])
    """
    return dT(n - 2, x) / S(2) - (dT(Abs(n - 4), x) + dT(n, x)) / S(4)


###########################
# Second Derivatives
###########################


def ddT(n, x):
    """
    >>> def _(n): return ddT(n, xT(n, np.arange(n)))
    >>> _(2)
    array([ 4.,  4.])
    >>> _(4)
    array([ 65.941,  -1.941,  -1.941,  65.941])
    >>> ddT(4, np.array([-1, 1]))
    array([ 80.,  80.])
    """
    return n * dU(n - 1, x)


def ddU(n, x):
    """
    >>> def _(n): return ddU(n, xU(n, np.arange(n)))
    >>> _(2)
    array([ 8.,  8.])
    >>> _(4)
    array([ 101.666,   -5.666,   -5.666,  101.666])
    >>> ddU(2, np.array([-1, +1]))
    array([ 8.,  8.])
    >>> ddU(3, np.array([-1, +1]))
    array([-48.,  48.])
    """

    def b(n): return (n - 1) * n * (n + 1) * (n + 2) * (n + 3) / 15

    return Piecewise(
        (b(n) * (-1) ** n, Eq(x, -1.0)),
        (b(n), Eq(x, +1.0)),
        ((n * (n + 2) * U(n, x) - 3 * x * dU(n, x)) / (x ** 2 - 1), True),
    )


def ddUclamp(n, x):
    """
    >>> def _(n): return ddUclamp(n, xUclamp(n, np.arange(n)))
    >>> _(2)
    array([-2., -2.])
    >>> _(4)
    array([-38.,  -2.,  -2., -38.])
    """
    return (ddT(n - 2, x) - ddT(n, x)) / 2


def ddTclamp(n, x):
    """
    >>> def _(n): return ddTclamp(n, xTclamp(n, np.arange(n)))
    >>> _(2)
    array([-2., -2.])
    >>> _(4)
    array([-18.,  -6.,  -6., -18.])
    """
    return ddT(n - 2, x) / S(2) - (ddT(Abs(n - 4), x) + ddT(n, x)) / S(4)


###########################
# Roots
###########################


def xT(N, n):
    """
    >>> for N in range(1, 4): print(xT(N, np.arange(N)).round(3) + .0)
    [ 0.]
    [-0.707  0.707]
    [-0.866  0.     0.866]
    >>> for N in range(1, 5): assert np.allclose(T(N, xT(N, np.arange(N))), 0)
    """
    return -cos(pi * (n + half) / N)


def xU(N, n):
    """
    >>> for N in range(1, 4): print(xU(N, np.arange(N)).round(3) + .0)
    [ 0.]
    [-0.5  0.5]
    [-0.707  0.     0.707]
    >>> for N in range(1, 5): assert np.allclose(U(N, xU(N, np.arange(N))), 0)
    """
    return -cos(pi * (n + 1) / (N + 1))


def xUclamp(N, n):
    """
    >>> for N in range(2, 6): print(xUclamp(N, np.arange(N)).round(3) + .0)
    [-1.  1.]
    [-1.  0.  1.]
    [-1.  -0.5  0.5  1. ]
    [-1.    -0.707  0.     0.707  1.   ]
    >>> for N in range(2, 6): assert np.allclose(Uclamp(N, xUclamp(N, np.arange(N))), 0)
    """
    return Piecewise(
        (-1, n <= 0),
        (+1, n >= N - 1),
        (-cos(pi * n / (N - 1)), True),
    )


def xTclamp(N, n):
    """
    >>> for N in range(2, 6): print(xTclamp(N, np.arange(N)).round(3) + .0)
    [-1.  1.]
    [-1.  0.  1.]
    [-1.    -0.707  0.707  1.   ]
    [-1.    -0.866  0.     0.866  1.   ]
    >>> for N in range(2, 6): assert np.allclose(Tclamp(N, xTclamp(N, np.arange(N))), 0)
    """
    n = np.array(n)
    return Piecewise(
        (-1, n <= 0),
        (+1, n >= N - 1),
        (-cos(pi * (n - half) / (N - 2)), True),
    )


###########################
# Vertex Cardinal Functions
###########################


def C(P, dP, xP):
    """
    >>> N = 3
    >>> m = np.arange(N)[np.newaxis]
    >>> n = m.T
    >>> ε = 1e-10
    >>> CT(N, m, xT(N, n)) + ε
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> CU(N, m, xU(N, n)) + ε
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> CUclamp(N, m, xUclamp(N, n)) + ε
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> CTclamp(N, m, xTclamp(N, n)) + ε
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> CT(N, np.arange(N), -1) + ε
    array([ 1.244, -0.333,  0.089])
    >>> CU(N, np.arange(N), -1) + ε
    array([ 1.707, -1.   ,  0.293])
    >>> CUclamp(N, np.arange(N), -1) + ε
    array([ 1.,  0.,  0.])
    >>> CTclamp(N, np.arange(N), -1) + ε
    array([ 1.,  0.,  0.])
    """

    def _(N, n, x):
        xn = xP(N, n)
        return Piecewise(
            (1, Eq(x, xn)),
            (P(N, x) / (x - xn) / dP(N, xn), True)
        )

    return _


CT = C(T, dT, xT)
CU = C(U, dU, xU)
CUclamp = C(Uclamp, dUclamp, xUclamp)
CTclamp = C(Tclamp, dTclamp, xTclamp)


###########################################
# Derivative of Vertex Cardinal Functions
###########################################

def dC(P, dP, ddP, xP):
    """
    >>> N = 3
    >>> m = np.arange(N)[np.newaxis]
    >>> n = m.T
    >>> ε = 1e-10
    >>> dCT(N, m, xT(N, n)) + ε
    array([[-1.732,  2.309, -0.577],
           [-0.577,  0.   ,  0.577],
           [ 0.577, -2.309,  1.732]])
    >>> dCU(N, m, xU(N, n)) + ε
    array([[-2.121,  2.828, -0.707],
           [-0.707,  0.   ,  0.707],
           [ 0.707, -2.828,  2.121]])
    >>> dCUclamp(N, m, xUclamp(N, n)) + ε
    array([[-1.5,  2. , -0.5],
           [-0.5,  0. ,  0.5],
           [ 0.5, -2. ,  1.5]])
    >>> dCTclamp(N, m, xTclamp(N, n)) + ε
    array([[-1.5,  2. , -0.5],
           [-0.5,  0. ,  0.5],
           [ 0.5, -2. ,  1.5]])
    >>> dCTclamp(5, 0, np.linspace(-1, -1+.1, 8))
    array([-9.5  , -9.035, -8.581, -8.14 , -7.711, -7.293, -6.887, -6.492])
    """

    def _(N, n, x):
        xn = xP(N, n)
        return Piecewise(
            (ddP(N, xn) / 2 / dP(N, xn), Eq(x, xn)),
            ((dP(N, x) * (x - xn) - P(N, x)) / (x - xn) ** 2 / dP(N, xn), True),
        )

    return _


dCT = dC(T, dT, ddT, xT)
dCU = dC(U, dU, ddU, xU)
dCUclamp = dC(Uclamp, dUclamp, ddUclamp, xUclamp)
dCTclamp = dC(Tclamp, dTclamp, ddTclamp, xTclamp)


###########################
# Edge Cardinal Functions
###########################

def D(dCP):
    """
    >>> DTclamp(5, 0, np.linspace(-1, 1, 8))
    array([ 9.5  ,  2.334, -1.054, -1.783, -0.972,  0.258,  0.789, -0.5  ])
    >>> DTclamp(5, 3, np.linspace(-1, 1, 8))
    array([-0.5  ,  0.789,  0.258, -0.972, -1.783, -1.054,  2.334,  9.5  ])

    >>> DUclamp(5, 0, np.linspace(-1, 1, 8))
    array([ 5.5  ,  1.774, -0.063, -0.57 , -0.308,  0.165,  0.287, -0.5  ])
    >>> DUclamp(5, 3, np.linspace(-1, 1, 8))
    array([-0.5  ,  0.287,  0.165, -0.308, -0.57 , -0.063,  1.774,  5.5  ])
    """

    def _(N, m, x):
        return -sum(dCP(N, n, x) for n in range(m + 1))

    return _


DT = D(dCT)
DU = D(dCU)
DUclamp = D(dCUclamp)
DTclamp = D(dCTclamp)


######################################
# Normalized Edge Cardinal Functions
######################################

def D(DP, xP):
    def _(N, n, x):
        return DP(N, n, x) * (xP(N, n + 1) - xP(N, n))

    return _


DnT = D(DT, xT)
DnU = D(DU, xU)
DnUclamp = D(DUclamp, xUclamp)
DnTclamp = D(DTclamp, xTclamp)
