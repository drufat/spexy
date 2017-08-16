# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np
from numpy.testing import assert_array_almost_equal as eq

from spexy.form.sym import Form
from spexy.grid.dim1.grid import Grid_1D
from spexy.grid.grid import Π
from sympy import sin, cos, symbols

x, = symbols('x,')
F0, F1 = Form.forms(x)


def test_refine():
    def check(G, N, f):
        """
        Avoid mixing float32 and float64 or else we will get mismatch
        between the values below.
        """

        g = G(N)
        points = G(2 * N).verts()[0]

        def _(g):
            T = g.dec.Pup(f.degree)
            U = g.dec.Pdown(f.degree)

            # discrete form
            fδ = g.Ps(f)

            # function form
            fλ = f.lambdify()

            # Test refinement (T)
            eq(fλ(points)[0], T(fδ.array))

            # Test unrefinement (U)
            eq(fδ.array, U(fλ(points)[0]))

        _(g)
        _(g.dual)

    g = Grid_1D.periodic
    check(g, 4, F0(sin(x)))
    check(g, 4, F1(sin(x)))

    for func in (sin(x), cos(x), sin(2 * x), sin(x) + cos(x), 1 + sin(x)):
        check(g, 5, F0(func))
        check(g, 5, F1(func))

    g = Grid_1D.chebyshev
    for func in (x, (1 - x) ** 2, x + 1):
        check(g, 4, F0(func))
        check(g, 4, F1(func))

    g = Grid_1D.chebyshev
    for func in (x, (1 - x) ** 2, x ** 3, x + 1):
        check(g, 5, F0(func))
        check(g, 5, F1(func))

    g = Grid_1D.chebnew
    for func in (x, (1 - x) ** 2, x + 1):
        check(g, 4, F0(func))
        check(g, 4, F1(func))

    g = Grid_1D.chebnew
    for func in (x, (1 - x) ** 2, x + 1):
        check(g, 5, F0(func))
        check(g, 5, F1(func))


def test_P():
    def _(g, f):
        assert g.Ps(f) == g.P(f)
        assert g.dual.Ps(f) == g.dual.P(f)

    g = Grid_1D.periodic(11)
    f = F0(sin(x))
    _(g, f)
    f = F1(cos(x))
    _(g, f)
    f = F1(cos(x) ** 3)
    _(g, f)

    g = Grid_1D.regular(11)
    f = F0(x)
    _(g, f)
    f = F1(x ** 3)
    _(g, f)

    for g in [
        Grid_1D.chebyshev(11),
        Grid_1D.chebold(11),
        Grid_1D.chebnew(11),
    ]:
        f = F0(x)
        _(g, f)
        f = F1(x ** 3)
        _(g, f)


def test_N():
    def _(g):
        f = F0(0)
        assert g.Ps(f).array.shape[0] == g.N[0]
        assert g.dual.Ps(f).array.shape[0] == g.dual.N[0]
        f = F1(0)
        assert g.Ps(f).array.shape[0] == g.N[1]
        assert g.dual.Ps(f).array.shape[0] == g.dual.N[1]

    _(Grid_1D.periodic(4))
    _(Grid_1D.chebyshev(3))
    _(Grid_1D.chebold(3))
    _(Grid_1D.chebnew(3))


def test_R():
    def _(g, f):
        pnts = np.linspace(g.xmin, g.xmax, 50)
        eq(f.lambdify()(pnts)[0], g.Ps(f).R(pnts))
        eq(f.lambdify()(pnts)[0], g.dual.Ps(f).R(pnts))

    g = Grid_1D.periodic(11)
    f = F0(sin(x))
    _(g, f)
    f = F1(sin(x))
    _(g, f)

    for g in [
        # can include boundaries, because native no longer gives nan there
        Grid_1D.chebyshev(11),
        Grid_1D.chebold(11),
        Grid_1D.chebnew(11),
    ]:
        f = F0(x ** 2)
        _(g, f)
        f = F1(x ** 2)
        _(g, f)


def test_D():
    def _(g, f):
        f = F0(f)

        assert g.P(f.D) == g.P(f).D + g.BC(f)
        assert g.dual.P(f.D) == g.dual.P(f).D + g.dual.BC(f)

        assert g.Ps(f.D) == g.Ps(f).D + g.BCs(f)
        assert g.dual.Ps(f.D) == g.dual.Ps(f).D + g.dual.BCs(f)

    for n in range(2, 6):
        for g in [
            Grid_1D.periodic(n),
        ]:
            _(g, cos(x))
            _(g, sin(x))

        for g in [
            Grid_1D.regular(n),
            Grid_1D.chebyshev(n),
            Grid_1D.chebold(n),
            Grid_1D.chebnew(n),
        ]:
            _(g, cos(x))
            _(g, sin(x))
            _(g, x)
            _(g, x ** 2)
            _(g, x ** 3)
            _(g, 1)


def test_H():
    def _(g, f):
        assert g.Ps(f.H) == g.dual.Ps(f).H
        assert g.dual.Ps(f.H) == g.Ps(f).H

    g = Grid_1D.periodic(11)
    f = F0(cos(3 * x))
    _(g, f)
    f = F1(sin(x) + cos(x))
    _(g, f)

    for g in [
        Grid_1D.chebyshev(11),
        Grid_1D.chebold(11),
        Grid_1D.chebnew(11),
    ]:
        f = F0(x ** 4)
        _(g, f)
        f = F1(x ** 2 + 1)
        _(g, f)


def test_W_C():
    def _(g, f0, f1):
        f = [f0, f1]
        for ((d1, g1), (d2, g2), g3) in Π(
                Π((0, 1), (g, g.dual)),
                Π((0, 1), (g, g.dual)),
                (g, g.dual)
        ):
            if d1 + d2 > g.dimension:
                continue
            assert g3.Ps(f[d1] ^ f[d2]) == g1.Ps(f[d1]).W(g2.Ps(f[d2]), to=g3)

        d1 = 1
        for (g1, (d2, g2), g3) in Π(
                (g, g.dual),
                Π((0, 1), (g, g.dual)),
                (g, g.dual)
        ):
            if d2 < 1:
                continue
            assert g3.Ps(f[d1] | f[d2]) == g1.Ps(f[d1]).C(g2.Ps(f[d2]), to=g3)

    _(
        Grid_1D.periodic(3),
        F0(cos(x)),
        F1(sin(x)),
    )
    _(
        Grid_1D.periodic(4),
        F0(cos(x)),
        F1(sin(x)),
    )
    _(
        Grid_1D.chebyshev(4),
        F0(x + 1),
        F1(x ** 2),
    )
    _(
        Grid_1D.chebyshev(5),
        F0(x + 1),
        F1(x ** 2),
    )
    _(
        Grid_1D.chebnew(4),
        F0(x + 1),
        F1(x ** 2),
    )
    _(
        Grid_1D.chebnew(5),
        F0(x + 1),
        F1(x ** 2),
    )
    _(
        Grid_1D.chebnew(5),
        F0(x - 0.2),
        F1((x + 0.1) ** 2),
    )
