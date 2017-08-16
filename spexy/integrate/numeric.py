# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#######################
# Numeric Intergration
#######################

import numpy as np
from scipy.integrate import quad, dblquad


def vectorize(f):
    return np.vectorize(f, excluded=[0])


def integrate(f, x0, x1):
    return quad(f, x0, x1)[0]


def integration_1d():
    """
    >>> P0, P1  = integration_1d()
    >>> P1(lambda x: x)(x0=0, x1=1)
    0.5
    >>> P1(lambda x: 1)(x0=0, x1=1)
    1.0
    >>> P1(lambda x: x**2)(x0=0, x1=1)
    0.33333333333333337
    """

    def P0(f):
        return lambda x0: f(x0)

    def P1(f):
        return lambda x0, x1: integrate(f, x0, x1)

    return P0, P1


def integration_2d():
    """
    >>> P0, P1, P2  = integration_2d()
    >>> P1(lambda x, y: (1, 0))(x0=0, y0=0, x1=1, y1=0)
    1.0
    >>> P2(lambda x, y: 1)(x0=0, y0=0, x1=1, y1=0, x2=1, y2=1)
    0.5

    (x2,y2)
       |\
       | \
       |  \
       |   \
       |    \
       |     \
       |      \
    (x0,y0)----(x1,y1)

    """

    def P0(f):
        def _(x0, y0):
            return f(x0, y0)

        return _

    def P1(f):
        def _(x0, y0, x1, y1):
            lx, ly = x1 - x0, y1 - y0

            def integrand(s):
                ux, uy = f(x0 * (1 - s) + x1 * s,
                           y0 * (1 - s) + y1 * s)
                return ux * lx + uy * ly

            return quad(integrand, 0, 1)[0]

        return _

    def P2(f):
        def _(x0, y0, x1, y1, x2, y2):
            A = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)

            def integrand(s, t):
                omega = f(x0 * (1 - s - t) + x1 * s + x2 * t,
                          y0 * (1 - s - t) + y1 * s + y2 * t)
                return omega * A

            return dblquad(integrand, 0, 1, lambda s: 0, lambda s: 1 - s)[0]

        return _

    return P0, P1, P2


def integration_2d_regular():
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

    >>> P0, P1, P2  = integration_2d_regular()
    >>> P1(lambda x, y: (1, 0))[0](x0=0, x1=1, y=0)
    1.0
    >>> P1(lambda x, y: (0, 1))[1](x=0, y0=0, y1=1)
    1.0
    >>> P2(lambda x, y: (1,))[0](x0=0, x1=1, y0=0, y1=1)
    1.0
    >>> P2(lambda x, y: (x,))[0](x0=0, x1=1, y0=0, y1=1)
    0.5
    """

    pick = (
        lambda f: lambda x, y: f(x, y)[0],
        lambda f: lambda x, y: f(x, y)[1],
    )

    def P0(f):
        f_ = pick[0](f)
        return (
            lambda x0, y0: f_(x0, y0),
        )

    def P1(f):
        fx = pick[0](f)
        fy = pick[1](f)
        return (
            lambda x0, x1, y: quad(lambda _: fx(_, y), x0, x1)[0],
            lambda x, y0, y1: quad(lambda _: fy(x, _), y0, y1)[0],
        )

    def P2(f):
        f_ = pick[0](f)
        return (
            lambda x0, x1, y0, y1: dblquad(f_, y0, y1, lambda y: x0, lambda y: x1)[0],
        )

    return P0, P1, P2
