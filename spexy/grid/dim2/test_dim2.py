# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np
from numpy import (sin, cos, exp)
from numpy.testing import assert_array_almost_equal as eq

from spexy.grid import Grid_2D


def test_basis_functions():
    def _(g):
        for P, B, N in zip(g.projection(), g.basis_fn(), g.numbers()):
            eq(np.vstack(P(B(i)) for i in range(N)), np.eye(N))

    for n in (1, 2):
        _(Grid_2D.periodic(n, n))
        _(Grid_2D.chebyshev(n, n))


def test_d0():
    def _(g, f, df):
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        D0, D1, D2, D0d, D1d, D2d = g.derivative()
        bc0, bc1, bc2, bc0d, bc1d, bc2d = g.boundary_cond()
        eq(D0(P0(f)) + bc1(f), P1(df))
        eq(D0d(P0d(f)) + bc1d(f), P1d(df))

    _(
        Grid_2D.periodic(4, 3),
        lambda x, y: (sin(x),),
        lambda x, y: (cos(x), 0)
    )
    _(
        Grid_2D.periodic(3, 5),
        lambda x, y: (sin(x) * cos(y),),
        lambda x, y: (cos(x) * cos(y), -sin(x) * sin(y))
    )
    _(
        Grid_2D.chebyshev(3, 5),
        lambda x, y: (x * y,),
        lambda x, y: (y, x)
    )
    _(
        Grid_2D.chebyshev(3, 3),
        lambda x, y: (x,),
        lambda x, y: (1, 0)
    )
    _(
        Grid_2D.chebnew(3, 5),
        lambda x, y: (x * y,),
        lambda x, y: (y, x)
    )
    _(
        Grid_2D.chebnew(3, 3),
        lambda x, y: (x,),
        lambda x, y: (1, 0)
    )


def test_d1():
    def _(g, f, df):
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        D0, D1, D2, D0d, D1d, D2d = g.derivative()
        bc0, bc1, bc2, bc0d, bc1d, bc2d = g.boundary_cond()
        eq(D1(P1(f)) + bc2(f), P2(df))
        eq(D1d(P1d(f)) + bc2d(f), P2d(df))

    _(
        Grid_2D.periodic(5, 6),
        lambda x, y: (sin(x), sin(y)),
        lambda x, y: (0,)
    )
    _(
        Grid_2D.periodic(5, 6),
        lambda x, y: (-sin(y), sin(x)),
        lambda x, y: (cos(x) + cos(y),)
    )
    _(
        Grid_2D.chebyshev(3, 5),
        lambda x, y: (-sin(y), sin(x)),
        lambda x, y: (cos(x) + cos(y),)
    )
    _(
        Grid_2D.chebyshev(3, 5),
        lambda x, y: (exp(-y), exp(x + y)),
        lambda x, y: (exp(-y) + exp(x + y),)
    )
    _(
        Grid_2D.chebnew(3, 5),
        lambda x, y: (-sin(y), sin(x)),
        lambda x, y: (cos(x) + cos(y),)
    )
    _(
        Grid_2D.chebnew(3, 5),
        lambda x, y: (exp(-y), exp(x + y)),
        lambda x, y: (exp(-y) + exp(x + y),)
    )


def test_h0():
    def _(g, f):
        H0, H1, H2, H0d, H1d, H2d = g.hodge_star()
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        eq(H0(P0(f)), P2d(f))
        eq(H2(P2(f)), P0d(f))
        eq(H0d(P0d(f)), P2(f))
        eq(H2d(P2d(f)), P0(f))

    _(
        Grid_2D.chebyshev(6, 6),
        lambda x, y: (x * y,)
    )
    _(
        Grid_2D.periodic(3, 5),
        lambda x, y: (sin(x) * sin(y),)
    )
    _(
        Grid_2D.chebyshev(5, 4),
        lambda x, y: (x * y,)
    )
    _(
        Grid_2D.chebnew(2, 3),
        lambda x, y: (x * y,)
    )


def test_h1():
    def _(g, u, v):
        H0, H1, H2, H0d, H1d, H2d = g.hodge_star()
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        eq(
            H1(P1(lambda x, y: (u(x, y), v(x, y)))),
            P1d(lambda x, y: (-v(x, y), u(x, y)))
        )
        eq(
            H1d(P1d(lambda x, y: (-v(x, y), u(x, y)))),
            P1(lambda x, y: (-u(x, y), -v(x, y)))
        )

    _(
        Grid_2D.periodic(3, 5),
        lambda x, y: sin(x) * sin(y),
        lambda x, y: cos(y)
    )
    _(
        Grid_2D.chebyshev(6, 6),
        lambda x, y: x * y,
        lambda x, y: y ** 2 - x
    )
    _(
        Grid_2D.chebyshev(3, 4),
        lambda x, y: x * y,
        lambda x, y: y ** 2 - x
    )
    _(
        Grid_2D.chebnew(3, 4),
        lambda x, y: x * y,
        lambda x, y: y ** 2 - x
    )
