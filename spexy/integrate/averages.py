# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import sympy as sy

from spexy.integrate.symbolic import process

###################
# Averages
###################

x, y, x0, y0, x1, y1, x2, y2 = sy.symbols('x, y, x0, y0, x1, y1, x2, y2')


def averages_1d(x, x0, x1):
    """
    >>> A0, A1 = averages_1d(x, x0, x1)
    >>> assert A0(1)     == 1
    >>> assert A0(x)     == x0
    >>> assert A1(1)     == 1
    >>> assert A1(x)     == (x1 + x0)/2
    >>> assert A1(x**2)  == (x0**2 + x0*x1 + x1**2)/3
    >>> assert A1(x**3)  == (x0**3 + x0**2*x1 + x0*x1**2 + x1**3)/4
    """
    s, t = sy.symbols('s t')
    assert t != x != s

    def A0(f):
        f = sy.sympify(f)
        return f.subs(x, x0)

    def A1(f):
        f = sy.sympify(f)
        integrand = f.subs(x, x0 * (1 - s) + x1 * s)
        iexpr = sy.Integral(integrand, (s, 0, 1))
        return process(iexpr)

    return A0, A1


def averages_2d(x, y, x0, y0, x1, y1, x2, y2):
    """
    >>> A0, A1, A2 = averages_2d(x, y, x0, y0, x1, y1, x2, y2)
    >>> assert A0(1) == 1
    >>> assert A0(x) == x0
    >>> assert A1(1) == 1
    >>> assert A1(x) == (x0 + x1)/2
    >>> assert A1(y) == (y0 + y1)/2
    >>> assert A2(1) == 1
    >>> assert A2(x) == (x0 + x1 + x2)/3
    >>> assert A2(y) == (y0 + y1 + y2)/3
    """
    s, t = sy.symbols('s t')
    assert t != x != s
    assert t != y != s

    def A0(f):
        f = sy.sympify(f)
        return f.subs({x: x0, y: y0})

    def A1(f):
        f = sy.sympify(f)
        subst = ((x, x0 * (1 - s) + x1 * s),
                 (y, y0 * (1 - s) + y1 * s))
        integrand = f.subs(subst)
        iexpr = sy.Integral(integrand, (s, 0, 1))
        return process(iexpr)

    def A2(f):
        f = sy.sympify(f)
        subst = ((x, x0 * (1 - s - t) + x1 * s + x2 * t),
                 (y, y0 * (1 - s - t) + y1 * s + y2 * t))
        integrand = 2 * f.subs(subst)
        iexpr = sy.Integral(integrand, (t, 0, 1 - s), (s, 0, 1))
        return process(iexpr)

    return A0, A1, A2
