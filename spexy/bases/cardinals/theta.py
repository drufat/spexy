# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from sympy import (acos, cos, sin, symbols, diff, Rational, limit, pi, Piecewise, Eq)
from spexy.symbolic import replace_piecewise_default

x, theta = symbols('x, theta')
N, n, m = symbols('N, n, m', integer=True)
half = Rational(1, 2)


def Theta(x):
    """
    >>> diff(Theta(x), x)
    1/sqrt(-x**2 + 1)
    >>> diff(diff(Theta(x), x), x)
    x/(-x**2 + 1)**(3/2)
    """
    return -acos(x)


def T(n, theta):
    return cos(n * theta)


def U(n, theta):
    """
    >>> u = replace_piecewise_default(U(n, theta))
    >>> (limit(u, theta, 0) - (n + 1)).simplify()
    0
    >>> (limit(u, theta, pi) - (n + 1)*(-1)**n).simplify()
    0
    """
    # return Piecewise(
    #     (n + 1, Eq(cos(theta), +1)),
    #     ((n + 1) * (-1) ** n, Eq(cos(theta), -1)),
    #     (sin((n + 1) * theta) / sin(theta), True)
    # )
    return sin((n + 1) * theta) / sin(theta)


def V(n, theta):
    """
    >>> (V(n, theta) - U(n-2, theta)*sin(theta)**2).simplify()
    0
    >>> (V(n, Theta(x)) - U(n-2, Theta(x))*(1-x**2)).simplify()
    0
    """
    return (T(n - 2, theta) - T(n, theta)) / 2


def dT(n, theta):
    """
    >>> assert dT(n, theta) == diff(T(n, theta), theta)
    """
    return -n * sin(n * theta)


def dU(n, theta):
    """
    >>> (dU(n, theta) - diff(U(n, theta), theta)).simplify()
    0
    """
    return (n * sin(theta) * cos((n + 1) * theta) - sin(n * theta)) / sin(theta) ** 2


def dV(n, theta):
    """
    >>> (dV(n, theta) - diff(V(n, theta), theta)).simplify()
    0
    """
    return n * sin(n * theta) / 2 - (n - 2) * sin(theta * (n - 2)) / 2


def ddT(n, theta):
    """
    >>> assert ddT(n, theta) == diff(dT(n, theta), theta)
    """
    return -n ** 2 * cos(n * theta)


def ddV(n, theta):
    """
    >>> (ddV(n, theta) - diff(dV(n, theta), theta)).simplify()
    0
    """
    return n ** 2 * cos(n * theta) / 2 - (n - 2) ** 2 * cos(theta * (n - 2)) / 2
