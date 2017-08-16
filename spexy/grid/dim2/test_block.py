# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np
from numpy import (sin, cos, exp)
from numpy.testing import assert_array_almost_equal as eq

from spexy.grid.dim2.grid_block import GridBlock_2D, eqv
from spexy.strides import gen


def test_refine0():
    def _(g, f):
        gg = g.refine()
        for g in (g, g.dual):
            assert eqv(
                g.dec.Pup(0)(g.dec.P(0)(f)),
                [
                    gg.dec.P(0)(f),
                ]
            )
            assert eqv(
                g.dec.P(0)(f),
                g.dec.Pdown(0)([
                    gg.dec.P(0)(f),
                ])
            )

    _(
        GridBlock_2D.periodic(3, 3),
        lambda x, y: (sin(x) + cos(y),)
    )
    _(
        GridBlock_2D.chebyshev(3, 4),
        lambda x, y: (x + y,)
    )
    _(
        GridBlock_2D.chebnew(3, 4),
        lambda x, y: (x + y,)
    )


def test_refine1():
    def _(g, f):
        gg = g.refine()
        ff = [
            lambda x, y: (f(x, y)[0],),
            lambda x, y: (f(x, y)[1],)
        ]

        for g in (g, g.dual):
            assert eqv(
                g.dec.Pup(1)(g.dec.P(1)(f)),
                [
                    gg.dec.P(0)(ff[0]),
                    gg.dec.P(0)(ff[1]),
                ],
            )
            assert eqv(
                g.dec.P(1)(f),
                g.dec.Pdown(1)([
                    gg.dec.P(0)(ff[0]),
                    gg.dec.P(0)(ff[1]),
                ])
            )

    _(
        GridBlock_2D.periodic(3, 3),
        lambda x, y: (-sin(y), cos(x))
    )
    _(
        GridBlock_2D.chebyshev(3, 4),
        lambda x, y: (-y, x)
    )
    _(
        GridBlock_2D.chebyshev(2, 2),
        lambda x, y: (x, y)
    )
    _(
        GridBlock_2D.chebnew(3, 4),
        lambda x, y: (-y, x)
    )
    _(
        GridBlock_2D.chebnew(4, 4),
        lambda x, y: (x, y)
    )


def test_basis_functions():
    def _(g):
        for P, B, N in zip(g.projection(), g.basis_fn(), g.numbers()):
            for k, i, j in gen(N):
                a = np.array([np.zeros(n) for n in N])
                a[k][i, j] = 1
                b = P(B[k](i, j))
                assert eqv(a, b)

    for n in (1, 2):
        _(GridBlock_2D.periodic(n, n))
        _(GridBlock_2D.chebyshev(n, n))
        _(GridBlock_2D.chebnew(n, n))


def test_projection_reconstruction():
    def _(P, R, N, xmin, xmax, ymin, ymax):
        a = np.array([np.random.rand(*n) for n in N])
        Ra = R(a)
        PRa = P(Ra)
        RPRa = R(PRa)

        assert eqv(a, PRa)
        for x in np.linspace(.8 * xmin, .8 * xmax, 3):
            for y in np.linspace(.8 * ymin, .8 * ymax, 3):
                eq(Ra(x, y), RPRa(x, y))

    def check_grid(g):
        for g in (g, g.dual):
            for k in range(g.dimension):
                _(g.proj_num(k), g.reconst(k), g.N[k],
                  g.gx.xmin, g.gx.xmax,
                  g.gy.xmin, g.gy.xmax)

    for n in (2, 3):
        check_grid(GridBlock_2D.periodic(n, n))
        check_grid(GridBlock_2D.chebyshev(n, n))
        check_grid(GridBlock_2D.chebnew(n, n))


def test_d0():
    def _(g, f, df):
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        D0, D1, D2, D0d, D1d, D2d = g.derivative()
        bc0, bc1, bc2, bc0d, bc1d, bc2d = g.boundary_cond()
        assert eqv(D0(P0(f)) + bc1(f), P1(df))
        assert eqv(D0d(P0d(f)) + bc1d(f), P1d(df))

    _(
        GridBlock_2D.periodic(4, 3),
        lambda x, y: (sin(x),),
        lambda x, y: (cos(x), 0)
    )

    _(
        GridBlock_2D.periodic(3, 5),
        lambda x, y: (sin(x) * cos(y),),
        lambda x, y: (cos(x) * cos(y), -sin(x) * sin(y))
    )

    _(
        GridBlock_2D.chebyshev(3, 5),
        lambda x, y: (x * y,),
        lambda x, y: (y, x)
    )

    _(
        GridBlock_2D.chebyshev(3, 3),
        lambda x, y: (x,),
        lambda x, y: (1, 0)
    )


def test_d1():
    def _(g, f, df):
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        D0, D1, D2, D0d, D1d, D2d = g.derivative()
        bc0, bc1, bc2, bc0d, bc1d, bc2d = g.boundary_cond()
        assert eqv(D1(P1(f)) + bc2(f), P2(df))
        assert eqv(D1d(P1d(f)) + bc2d(f), P2d(df))

    _(
        GridBlock_2D.periodic(5, 6),
        lambda x, y: (sin(x), sin(y)),
        lambda x, y: (0,)
    )

    _(
        GridBlock_2D.periodic(5, 6),
        lambda x, y: (-sin(y), sin(x)),
        lambda x, y: (cos(x) + cos(y),)
    )

    _(
        GridBlock_2D.chebyshev(3, 5),
        lambda x, y: (-sin(y), sin(x)),
        lambda x, y: (cos(x) + cos(y),)
    )

    _(
        GridBlock_2D.chebyshev(3, 5),
        lambda x, y: (exp(-y), exp(x + y)),
        lambda x, y: (exp(-y) + exp(x + y),)
    )


def test_h0():
    def _(g, f):
        H0, H1, H2, H0d, H1d, H2d = g.hodge_star()
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        assert eqv(H0(P0(f)), P2d(f))
        assert eqv(H2(P2(f)), P0d(f))
        assert eqv(H0d(P0d(f)), P2(f))
        assert eqv(H2d(P2d(f)), P0(f))

    _(GridBlock_2D.chebyshev(6, 6),
      lambda x, y: (x * y,))

    _(GridBlock_2D.chebyshev(5, 4),
      lambda x, y: (x * y,))

    _(GridBlock_2D.periodic(3, 5),
      lambda x, y: (sin(x) * sin(y),))


def test_h1():
    def _(g, u, v):
        H0, H1, H2, H0d, H1d, H2d = g.hodge_star()
        P0, P1, P2, P0d, P1d, P2d = g.projection()
        assert eqv(
            H1(P1(lambda x, y: (u(x, y), v(x, y)))),
            P1d(lambda x, y: (-v(x, y), u(x, y)))
        )
        assert eqv(
            H1d(P1d(lambda x, y: (-v(x, y), u(x, y)))),
            P1(lambda x, y: (-u(x, y), -v(x, y)))
        )

    _(GridBlock_2D.periodic(3, 5),
      lambda x, y: sin(x) * sin(y),
      lambda x, y: cos(y))

    _(GridBlock_2D.chebyshev(6, 6),
      lambda x, y: x * y,
      lambda x, y: y ** 2 - x)

    _(GridBlock_2D.chebyshev(3, 4),
      lambda x, y: x * y,
      lambda x, y: y ** 2 - x)
