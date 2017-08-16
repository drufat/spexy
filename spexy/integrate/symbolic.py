# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
##########################
# Symbolic Integration
##########################

import sympy as sy
from spexy.data import memoize

x, y, x0, y0, x1, y1, x2, y2 = sy.symbols('x, y, x0, y0, x1, y1, x2, y2')


@memoize('memoize/sympy.json', lambda expr: repr(expr), lambda expr: sy.sympify(expr))
def process(expr):
    rslt = sy.simplify(expr.doit())
    if rslt.has(sy.Integral):
        raise ValueError('Unable to evaluate {}.'.format(rslt))
    return rslt


def integrate(f, x, a, b):
    return process(sy.Integral(f, (x, a, b)))


def integration_1d_sym(x, x0, x1):
    """
    >>> P0, P1 = integration_1d_sym(x, x0, x1)
    >>> assert P0(x) == x0
    >>> assert P1(x) == -x0**2/2 + x1**2/2
    >>> assert P1(1) == -x0 + x1
    """

    def P0(f):
        f = sy.sympify(f)
        return f.subs(x, x0)

    def P1(f):
        f = sy.sympify(f)
        return integrate(f, x, x0, x1)

    return P0, P1


def integration_1d():
    """
    >>> P0, P1  = integration_1d()
    >>> P1(lambda x: x)(0, 1)
    0.5
    >>> P1(lambda x: 1)(0, 1)
    1
    >>> P1(lambda x: x**2)(0, 1)
    0.3333333333333333
    """
    I = integration_1d_sym(x, x0, x1)

    def P0(f):
        p = I[0](f(x))
        return sy.lambdify((x0,), p)

    def P1(f):
        p = I[1](f(x))
        return sy.lambdify((x0, x1), p)

    return P0, P1


def integrate_2d_1form(fx, fy, x, y, x0, y0, x1, y1):
    s = sy.Symbol('s')
    lx, ly = x1 - x0, y1 - y0
    subst = ((x, x0 * (1 - s) + x1 * s),
             (y, y0 * (1 - s) + y1 * s))
    integrand = (fx.subs(subst) * lx +
                 fy.subs(subst) * ly)
    return integrate(integrand, s, 0, 1)


def integrate_2d_2form(f, x, y, x0, y0, x1, y1, x2, y2):
    s, t = sy.symbols('s, t')
    A = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    subst = ((x, x0 * (1 - s - t) + x1 * s + x2 * t),
             (y, y0 * (1 - s - t) + y1 * s + y2 * t))
    integrand = (f.subs(subst) * A)
    iexpr = sy.Integral(integrand, (t, 0, 1 - s), (s, 0, 1))
    return process(iexpr)


def integration_2d(x, y, x0, y0, x1, y1, x2, y2):
    """
    >>> P0, P1, P2 = integration_2d(x, y, x0, y0, x1, y1, x2, y2)
    >>> assert P0((x*y,)) == x0*y0
    >>> assert P1((x, 0)) == -x0**2/2 + x1**2/2
    >>> assert P1((1, 0)) == -x0 + x1
    >>> assert P1((1, 1)) == -x0 + x1 - y0 + y1
    >>> assert P2((0,)) == 0

    The expression below corresponds to the area of a triangle
    (x2,y2)
       |\
       | \
       |  \
       |   \
       |    \
       |     \
       |      \
    (x0,y0)----(x1,y1)
    >>> from sympy import expand
    >>> assert P2((1,)) == expand( ((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))/2 )
    """

    def P0(f):
        f, = sy.sympify(f)
        return f.subs({x: x0, y: y0})

    def P1(f):
        fx, fy = sy.sympify(f)
        return integrate_2d_1form(fx, fy, x, y, x0, y0, x1, y1)

    def P2(f):
        f, = sy.sympify(f)
        return integrate_2d_2form(f, x, y, x0, y0, x1, y1, x2, y2)

    return P0, P1, P2


def integration_2d_regular_sym(x, y, x0, y0, x1, y1):
    """
    Regular integration.
    P0, P1x, P1y, P2

    Horizontal edges.
    (x0,y)--------(x1,y)

    Vertical edges.
        (x,y1)
           |
           |
        (x,y0)

    Rectangles
    (x0,y1)--------(x1,y1)
       |              |
       |              |
    (x0,y0)--------(x1,y0)

    >>> P0, P1, P2 = integration_2d_regular_sym(x, y, x0, y0, x1, y1)
    >>> assert P0((1,)) == (1,)
    >>> assert P0((x*y,)) == (x0*y0,)
    >>> assert P1((1, 0)) == ((x1-x0), 0)
    >>> assert P1((0, 1)) == (0, (y1-y0))
    >>> assert P1((y, x)) == (y*(x1-x0), x*(y1-y0))
    >>> assert P2((0,)) == (0,)
    >>> assert P2((1,)) == ((x0-x1)*(y0-y1),)
    """

    def P0(f):
        f, = sy.sympify(f)
        f = f.subs({x: x0, y: y0})
        return (f,)

    def P1(f):
        fx, fy = sy.sympify(f)
        fx = integrate(fx, x, x0, x1)
        fy = integrate(fy, y, y0, y1)
        return (fx, fy)

    def P2(f):
        f, = sy.sympify(f)
        f = integrate(f, x, x0, x1)
        f = integrate(f, y, y0, y1)
        return (f,)

    return P0, P1, P2


def integration_2d_regular():
    """
    >>> P0, P1, P2  = integration_2d_regular()
    >>> P1(lambda x, y: (1, 0))[0](0, 1, 0)
    1
    >>> P1(lambda x, y: (0, 1))[1](0, 0, 1)
    1
    >>> P2(lambda x, y: (1,))[0](0, 1, 0, 1)
    1
    >>> P2(lambda x, y: (x,))[0](0, 1, 0, 1)
    0.5
    """
    I = integration_2d_regular_sym(x, y, x0, y0, x1, y1)

    def P0(f):
        p = I[0](f(x, y))
        return (
            sy.lambdify((x0, y0), p[0]),
        )

    def P1(f):
        p = I[1](f(x, y))
        return (
            sy.lambdify((x0, x1, y), p[0]),
            sy.lambdify((x, y0, y1), p[1]),
        )

    def P2(f):
        p = I[2](f(x, y))
        return (
            sy.lambdify((x0, x1, y0, y1), p[0]),
        )

    return P0, P1, P2
