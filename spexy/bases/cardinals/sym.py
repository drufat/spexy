# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import sympy as sy
from spexy.symbolic import replace_piecewise_default
from sympy import (cos, symbols, Rational, pi, Eq, poly, div, expand, Abs)

x, theta = symbols('x, theta')
N, n, m = symbols('N, n, m', integer=True)
S = sy.S
half = Rational(1, 2)


def diff(f, *symbols, **kwargs):
    return sy.diff(replace_piecewise_default(f), *symbols, **kwargs)


def simplify(expr):
    return sy.simplify(replace_piecewise_default(expr))


def trigsimp(expr):
    return sy.trigsimp(replace_piecewise_default(expr))


###########################
# Chebyshev Polynomials
###########################


def T(N, x):
    """
    >>> [S(T(n, x)).subs(x, 1) for n in range(10)]
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    >>> [S(T(n, x)).subs(x, -1) for n in range(10)]
    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    """
    if N == 0:
        return 1
    n, Tprev, T = 1, 1, x
    while n < N:
        n, Tprev, T = (n + 1), T, (2 * x * T - Tprev)
    return poly(T, x).args[0]


def U(N, x):
    """
    >>> [S(U(n, x)).subs(x, 1) for n in range(10)]
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> [S(U(n, x)).subs(x, -1) for n in range(10)]
    [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]
    """
    if N == -1:
        return 0
    if N == 0:
        return 1
    n, Uprev, U = 1, 1, 2 * x
    while n < N:
        n, Uprev, U = (n + 1), U, (2 * x * U - Uprev)
    return poly(U, x).args[0]


def Uclamp(n, x):
    """
    >>> Uclamp(2, x)
    -x**2 + 1
    >>> Uclamp(3, x)
    -2*x**3 + 2*x
    >>> def Uclamp_(n, x): return U(n - 2, x) * (1 - x**2)
    >>> [expand(Uclamp(n, x) - Uclamp_(n, x))  for n in range(2, 7)]
    [0, 0, 0, 0, 0]
    """
    return (T(n - 2, x) - T(n, x)) / S(2)


def Tclamp(n, x):
    """
    >>> Tclamp(2, x)
    -x**2 + 1
    >>> Tclamp(3, x)
    -x**3 + x
    >>> def Tclamp_(n, x): return T(n - 2, x) * (1 - x**2)
    >>> [expand(Tclamp(n, x) - Tclamp_(n, x))  for n in range(2, 8)]
    [0, 0, 0, 0, 0, 0]
    """
    return T(n - 2, x) / S(2) - (T(Abs(n - 4), x) + T(n, x)) / S(4)


###########################
# First Derivatives
###########################

def dT(n, x):
    """
    >>> [S(dT(n, x)).subs(x, 1) for n in range(10)]
    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    >>> [S(dT(n, x)).subs(x, -1) for n in range(10)]
    [0, 1, -4, 9, -16, 25, -36, 49, -64, 81]
    >>> dT(2, x)
    4*x
    >>> dT(3, x)
    12*x**2 - 3
    >>> [dT(n, x) - diff(T(n, x), x)  for n in range(8)]
    [0, 0, 0, 0, 0, 0, 0, 0]
    """
    return n * U(n - 1, x)


def dU(n, x):
    """
    >>> [dU(n, x).subs(x, 1) for n in range(10)]
    [0, 2, 8, 20, 40, 70, 112, 168, 240, 330]
    >>> [dU(n, x).subs(x, -1) for n in range(10)]
    [0, 2, -8, 20, -40, 70, -112, 168, -240, 330]
    >>> dU(-1, x)
    0
    >>> dU(0, x)
    0
    >>> dU(1, x)
    2
    >>> dU(2, x)
    8*x
    >>> dU(3, x)
    24*x**2 - 4
    >>> dU(10, x)
    10240*x**9 - 18432*x**7 + 10752*x**5 - 2240*x**3 + 120*x
    >>> [dU(n, x) - diff(U(n, x), x) for n in range(8)]
    [0, 0, 0, 0, 0, 0, 0, 0]
    """
    q, r = div(
        (n + 1) * T(n + 1, x) - x * U(n, x),
        x ** 2 - 1,
        x
    )
    return q


def dUclamp(n, x):
    """
    >>> dUclamp(2, x)
    -2*x
    >>> dUclamp(3, x)
    -6*x**2 + 2
    >>> [ dUclamp(n, x) - diff(Uclamp(n, x), x)  for n in range(2, 8)]
    [0, 0, 0, 0, 0, 0]
    """
    return (dT(n - 2, x) - dT(n, x)) / 2


def dTclamp(n, x):
    """
    >>> dTclamp(2, x)
    -2*x
    >>> dTclamp(3, x)
    -3*x**2 + 1
    >>> [ dTclamp(n, x) - diff(Tclamp(n, x), x)  for n in range(2, 8)]
    [0, 0, 0, 0, 0, 0]
    """
    return dT(n - 2, x) / S(2) - (dT(Abs(n - 4), x) + dT(n, x)) / S(4)


###########################
# Second Derivatives
###########################


def ddT(n, x):
    """
    >>> [ddT(n, x).subs(x, 1) for n in range(10)]
    [0, 0, 4, 24, 80, 200, 420, 784, 1344, 2160]
    >>> [ddT(n, x).subs(x, -1) for n in range(10)]
    [0, 0, 4, -24, 80, -200, 420, -784, 1344, -2160]
    >>> ddT(0, x)
    0
    >>> ddT(1, x)
    0
    >>> ddT(2, x)
    4
    >>> ddT(3, x)
    24*x
    >>> [ ddT(n, x) - diff(dT(n, x), x) for n in range(8)]
    [0, 0, 0, 0, 0, 0, 0, 0]
    """
    return n * dU(n - 1, x)


def ddU(n, x):
    """
    >>> [ddU(n, x).subs(x, 1) for n in range(10)]
    [0, 0, 8, 48, 168, 448, 1008, 2016, 3696, 6336]
    >>> [ddU(n, x).subs(x, -1) for n in range(10)]
    [0, 0, 8, -48, 168, -448, 1008, -2016, 3696, -6336]
    >>> ddU(0, x)
    0
    >>> ddU(1, x)
    0
    >>> ddU(2, x)
    8
    >>> ddU(3, x)
    48*x
    >>> [simplify( ddU(n, x) - diff(dU(n, x), x) ) for n in range(8)]
    [0, 0, 0, 0, 0, 0, 0, 0]
    """
    q, r = div(
        n * (n + 2) * U(n, x) - 3 * x * dU(n, x),
        x ** 2 - 1
    )
    return q


def ddUclamp(n, x):
    """
    >>> ddUclamp(2, x)
    -2
    >>> ddUclamp(3, x)
    -12*x
    >>> [simplify( ddUclamp(n, x) - diff(dUclamp(n, x), x) ) for n in range(2, 8)]
    [0, 0, 0, 0, 0, 0]
    """
    return (ddT(n - 2, x) - ddT(n, x)) / 2


def ddTclamp(n, x):
    """
    >>> ddTclamp(2, x)
    -2
    >>> ddTclamp(3, x)
    -6*x
    >>> [simplify( ddTclamp(n, x) - diff(dTclamp(n, x), x) ) for n in range(2, 8)]
    [0, 0, 0, 0, 0, 0]
    """
    return ddT(n - 2, x) / S(2) - (ddT(Abs(n - 4), x) + ddT(n, x)) / S(4)


###########################
# Roots
###########################


def xT(N, n):
    """
    >>> for N in range(1, 4): print([trigsimp( xT(N, n) ) for n in range(N)])
    [0]
    [-sqrt(2)/2, sqrt(2)/2]
    [-sqrt(3)/2, 0, sqrt(3)/2]
    >>> [[simplify( T(N, xT(N, n)) ) for n in range(N)] for N in range(1, 5)]
    [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0]]
    """
    return -cos(pi * (n + half) / N)


def xU(N, n):
    """
    >>> for N in range(1, 4): print([trigsimp( xU(N, n) ) for n in range(N)])
    [0]
    [-1/2, 1/2]
    [-sqrt(2)/2, 0, sqrt(2)/2]
    >>> [[simplify( U(N, xU(N, n)) ) for n in range(N)] for N in range(1, 5)]
    [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0]]
    """
    return -cos(pi * (n + 1) / (N + 1))


def xUclamp(N, n):
    """
    >>> for N in range(2, 6): print([trigsimp( xUclamp(N, n) ) for n in range(N)])
    [-1, 1]
    [-1, 0, 1]
    [-1, -1/2, 1/2, 1]
    [-1, -sqrt(2)/2, 0, sqrt(2)/2, 1]
    >>> [[simplify( Uclamp(N, xUclamp(N, n)) ) for n in range(N)] for N in range(2, 6)]
    [[0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0]]
    """
    if n <= 0:
        return S(-1)
    if n >= N - 1:
        return S(+1)
    return -cos(pi * n / (N - 1))


def xTclamp(N, n):
    """
    >>> for N in range(2, 6): print([trigsimp( xTclamp(N, n) ) for n in range(N)])
    [-1, 1]
    [-1, 0, 1]
    [-1, -sqrt(2)/2, sqrt(2)/2, 1]
    [-1, -sqrt(3)/2, 0, sqrt(3)/2, 1]
    """
    if n <= 0:
        return S(-1)
    if n >= N - 1:
        return S(+1)
    return -cos(pi * (n - half) / (N - 2))


###########################
# Vertex Cardinal Functions
###########################


def C(P, dP, xP):
    """
    >>> CT, CU, CUclamp = ((lambda CP: lambda *args: expand(CP(*args)))(CP) for CP in (CT, CU, CUclamp))
    >>> CT(1, 0, x)
    1
    >>> CU(1, 0, x)
    1
    >>> CT(2, 0, x)
    -sqrt(2)*x/2 + 1/2
    >>> CU(2, 0, x)
    -x + 1/2
    >>> CUclamp(2, 0, x)
    -x/2 + 1/2
    >>> CTclamp(2, 0, x)
    -x/2 + 1/2
    """

    def _(N, n, x):
        xn = xP(N, n)
        q, r = div(P(N, x), (x - xn), x)
        return q / dP(N, xn)

    return _


CT = C(T, dT, xT)
CU = C(U, dU, xU)
CUclamp = C(Uclamp, dUclamp, xUclamp)
CTclamp = C(Tclamp, dTclamp, xTclamp)


###########################################
# Derivative of Vertex Cardinal Functions
###########################################

def dC(P, dP, xP):
    """
    >>> dCT(1, 0, x)
    0
    >>> dCU(1, 0, x)
    0
    >>> dCT(2, 0, x)
    -sqrt(2)/2
    >>> [simplify(dCT(N, n, x) - diff(CT(N, n, x), x)) for N in range(2, 4) for n in range(N)]
    [0, 0, 0, 0, 0]
    >>> [simplify((dCU(N, n, x) - diff(CU(N, n, x), x))) for N in range(2, 4) for n in range(N)]
    [0, 0, 0, 0, 0]
    >>> [simplify((dCUclamp(N, n, x) - diff(CUclamp(N, n, x), x))) for N in range(2, 4) for n in range(N)]
    [0, 0, 0, 0, 0]
    """

    def _(N, n, x):
        xn = xP(N, n)
        q, r = div(dP(N, x) * (x - xn) - P(N, x), (x - xn) ** 2)
        return q / dP(N, xn)

    return _


dCT = dC(T, dT, xT)
dCU = dC(U, dU, xU)
dCUclamp = dC(Uclamp, dUclamp, xUclamp)
dCTclamp = dC(Tclamp, dTclamp, xTclamp)


###########################
# Edge Cardinal Functions
###########################

def D(dCP):
    """
    >>> DT(2, 0, x)
    sqrt(2)/2
    >>> DU(2, 0, x)
    1
    >>> DUclamp(2, 0, x)
    1/2
    >>> DTclamp(2, 0, x)
    1/2
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
