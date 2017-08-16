# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np

import sympy as sy
from spexy.form.sym import Form
from spexy.grid.dim2.grid import Grid_2D
from spexy.grid.grid import Π
from sympy import sin, cos, symbols

x, y = symbols('x, y')
F0, F1, F2 = Form.forms(x, y)


def test_sym():
    assert F0(x * y) == Form(0, (x, y), (x * y,))
    assert F1(x, y) == Form(1, (x, y), (x, y,))
    assert F2(x + y) == Form(2, (x, y), (x + y,))


def test_refine():
    def check(g, f):
        """
        Avoid mixing float32 and float64 or else we will get mismatch
        between the values below.
        """

        def _(g):
            points = g.refine().verts()
            T = g.upsample(f.degree)
            U = g.downsample(f.degree)

            # discrete form
            fδ = g.Ps(f)

            # function form
            fλ = sy.lambdify(f.grid.coord, f.comp, 'numpy')

            # Test refinement (T)
            assert np.allclose(fλ(*points), T(fδ.array))

            # Test unrefinement (U)
            assert np.allclose(fδ.array, U(fλ(*points)))

        _(g)
        _(g.dual)

    g = Grid_2D.periodic(3, 3)
    check(g, F0(sin(x) + cos(y)))
    check(g, F1(sin(x), sin(y)))
    check(g, F1(cos(x), cos(y)))
    check(g, F1(-sin(y), sin(x)))
    check(g, F1(-cos(y), cos(x)))
    check(g, F2(sin(x)))

    g = Grid_2D.chebyshev(3, 3)
    check(g, F0(x + y))
    check(g, F1(x, y))
    check(g, F1(-y, x))
    check(g, F2(x + y))
    check(g, F2(x))
    check(g, F2(y))

    g = Grid_2D.chebnew(4, 4)
    check(g, F0(x + y))
    check(g, F1(x, y))
    check(g, F1(-y, x))
    check(g, F2(x + y))
    check(g, F2(x))
    check(g, F2(y))


def test_N():
    def _(g):
        f = F0(0)
        assert g.Ps(f).array.shape[0] == g.N[0]
        assert g.dual.Ps(f).array.shape[0] == g.dual.N[0]
        f = F1(0, 0)
        assert g.Ps(f).array.shape[0] == g.N[1]
        assert g.dual.Ps(f).array.shape[0] == g.dual.N[1]
        f = F2(0)
        assert g.Ps(f).array.shape[0] == g.N[2]
        assert g.dual.Ps(f).array.shape[0] == g.dual.N[2]

    _(Grid_2D.periodic(4, 3))
    _(Grid_2D.periodic(3, 5))
    _(Grid_2D.chebyshev(2, 3))
    _(Grid_2D.chebyshev(5, 4))


def test_D():
    def _(g, f, bc=0):
        assert g.Ps(f.D) == g.Ps(f).D
        assert g.dual.Ps(f.D) == g.dual.Ps(f).D + bc

    g = Grid_2D.periodic(3, 3)

    f = F0(cos(x) + cos(y))
    _(g, f)

    f = F1(cos(x), cos(y))
    _(g, f)

    g = Grid_2D.chebyshev(3, 3)

    f = F0(x + y)
    _(g, f, g.dual.BCs(f).array)

    f = F1(-y, x)
    _(g, f, g.dual.BCs(f).array)

    g = Grid_2D.chebyshev(4, 5)

    f = F0(x * y)
    _(g, f, g.dual.BCs(f).array)

    f = F1(x, y)
    _(g, f, g.dual.BCs(f).array)


def test_H():
    def _(g, f):
        assert g.Ps(f.H) == g.dual.Ps(f).H
        assert g.dual.Ps(f.H) == g.Ps(f).H

    g = Grid_2D.periodic(5, 5)

    f = F0(cos(x))
    _(g, f)

    f = F1(sin(x), cos(y))
    _(g, f)

    g = Grid_2D.chebyshev(10, 10)

    f = F0(x ** 4)
    _(g, f)

    f = F1(-y, x ** 2)
    _(g, f)

    f = F2(x ** 2 * y)
    _(g, f)


def test_W_C():
    g = Grid_2D.chebyshev(4, 4)

    f0 = F0(x)
    f1 = F1(x ** 2, -y)
    f2 = F2(x + y)
    f = [f0, f1, f2]

    for ((d1, g1), (d2, g2), g3) in Π(
            Π((0, 1), (g, g.dual)),
            Π((0, 1), (g, g.dual)),
            (g, g.dual)
    ):
        if d1 + d2 > g.dimension:
            continue
        assert g3.Ps(f[d1] ^ f[d2]) == g1.Ps(f[d1]).W(g2.Ps(f[d2]), to=g3)

    h0 = F0(y)
    h1 = F1(-y, x)
    h2 = F2(x - y)
    h = [h0, h1, h2]

    d1 = 1
    for (g1, (d2, g2), g3) in Π(
            (g, g.dual),
            Π((0, 1), (g, g.dual)),
            (g, g.dual)
    ):
        if d2 < 1:
            continue
        assert g3.Ps(f[d1] | h[d2]) == g1.Ps(f[d1]).C(g2.Ps(h[d2]), to=g3)
