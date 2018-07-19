# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np
from spexy.grid.dim1.grid import Grid_1D
from spexy.grid.grid import hodge_star_matrix
from spexy.spectral import to_matrix, lagrange_polynomials
from numpy import (sin, cos, pi, sqrt)
from numpy.testing import assert_array_almost_equal as eq

np.random.seed(seed=1)


def test_basis_functions():
    def check_grid(g):
        for P, B, N in zip(g.projection(), g.basis_fn(), g.numbers()):
            eq(np.vstack(P(B(i)) for i in range(N)), np.eye(N))

    for n in range(3, 6):
        check_grid(Grid_1D.periodic(n))
        check_grid(Grid_1D.regular(n))
        check_grid(Grid_1D.chebyshev(n))
        check_grid(Grid_1D.chebnew(n))
        check_grid(Grid_1D.chebold(n))


def test_delta():
    g = Grid_1D.periodic(2)
    eq(np.fromfunction(g.delta[1], (2,)), [pi, pi])
    eq(np.fromfunction(g.dual.delta[1], (2,)), [pi, pi])

    g = Grid_1D.regular(2)
    eq(np.fromfunction(g.delta[1], (2,)), [pi / 2, pi / 2])
    eq(np.fromfunction(g.dual.delta[1], (3,)), [pi / 4, pi / 2, pi / 4])

    g = Grid_1D.chebyshev(2)
    eq(np.fromfunction(g.delta[1], (2,)), [1., 1.])
    eq(np.fromfunction(g.dual.delta[1], (1,)), [.292893])

    g = Grid_1D.chebold(2)
    eq(np.fromfunction(g.delta[1], (2,)), [1., 1.])
    eq(np.fromfunction(g.dual.delta[1], (1,)), [.292893])

    g = Grid_1D.chebnew(2)
    eq(np.fromfunction(g.delta[1], (2,)), [1., 1.])
    eq(np.fromfunction(g.dual.delta[1], (1,)), [sqrt(2)])


def test_projection_reconstruction():
    def _(g):
        for P, R, N in zip(g.projection(), g.reconstruction(), g.numbers()):
            y = np.random.rand(N)
            eq(P(R(y)), y)

    for n in range(2, 5):
        _(Grid_1D.periodic(n))
        _(Grid_1D.regular(n))
        _(Grid_1D.chebyshev(n))
        _(Grid_1D.chebnew(n))
        _(Grid_1D.chebold(n))


def test_hodge_star_basis_fn():
    def _(g):
        H0, H1, H0d, H1d = hodge_star_matrix(g)
        h0, h1, h0d, h1d = g.hodge_star()
        N0, N1, N0d, N1d = g.numbers()

        eq(H0d, to_matrix(h0d, N0d))
        eq(H1, to_matrix(h1, N1))

        eq(H0, to_matrix(h0, N0))
        eq(H1d, to_matrix(h1d, N1d))

    for n in range(2, 4):
        _(Grid_1D.periodic(n))
        _(Grid_1D.regular(n))
        _(Grid_1D.chebyshev(n))
        _(Grid_1D.chebnew(n))
        _(Grid_1D.chebold(n))


def test_hodge_star_inv():
    def _(g):
        h0, h1, h0d, h1d = g.hodge_star()
        n0, n1, n0d, n1d = g.numbers()

        H0 = to_matrix(h0, n0)
        H1d = to_matrix(h1d, n1d)
        eq(H1d, np.linalg.inv(H0))

        H1 = to_matrix(h1, n1)
        H0d = to_matrix(h0d, n0d)
        eq(H0d, np.linalg.inv(H1))

    for n in range(3, 7):
        _(Grid_1D.periodic(n))
        _(Grid_1D.regular(n))
        _(Grid_1D.chebyshev(n))
        _(Grid_1D.chebnew(n))
        _(Grid_1D.chebold(n))


def test_compare_chebyshev_and_lagrange_polynomials():
    """
    The Chebyshev basis functions are equivalent to the
    Lagrange basis functions for the grid points.
    """

    def _(g):
        x = np.linspace(g.xmin, g.xmax, 100)
        B0, B1, Bd0, Bd1 = g.basis_fn()

        L = lagrange_polynomials(*g.verts())
        for i in range(len(L)):
            eq(L[i](x), B0(i)(x))

        L = lagrange_polynomials(*g.dual.verts())
        for i in range(len(L)):
            eq(L[i](x), Bd0(i)(x))

    for n in range(3, 7):
        _(Grid_1D.chebyshev(n))
        _(Grid_1D.chebnew(n))
        _(Grid_1D.chebold(n))


def test_d():
    """
    >>> g = Grid_1D.periodic(3)
    >>> g.n
    3
    >>> from spexy.spectral import to_matrix
    >>> to_matrix(g.dec.D(0), g.N[0])
    array([[-1.,  1.,  0.],
           [ 0., -1.,  1.],
           [ 1.,  0., -1.]])
    >>> to_matrix(g.dual.dec.D(0), g.dual.N[0])
    array([[ 1.,  0., -1.],
           [-1.,  1.,  0.],
           [ 0., -1.,  1.]])
    """

    def _(g, f, f_prime):
        D0, D1, D0d, D1d = g.derivative()
        P0, P1, P0d, P1d = g.projection()
        bc0, bc1, bc0d, bc1d = g.boundary_cond()
        eq(D0(P0(f)) + bc1(f), P1(f_prime))
        eq(D0d(P0d(f)) + bc1d(f), P1d(f_prime))

    for g in (
            Grid_1D.periodic(10),
            Grid_1D.periodic(11),
    ):
        _(g,
          lambda x: sin(x),
          lambda x: cos(x))
    for g in (
            Grid_1D.chebyshev(10),
            Grid_1D.chebyshev(11),
            Grid_1D.chebnew(10),
            Grid_1D.chebnew(11),
    ):
        _(g,
          lambda x: 2 * x,
          lambda x: 2 + 0 * x)
        _(g,
          lambda x: x ** 3,
          lambda x: 3 * x ** 2)


def test_hodge():
    def _(g, f):
        assert g.P(0, f).H == g.dual.P(1, f)
        assert g.P(1, f).H == g.dual.P(0, f)

    _(Grid_1D.periodic(10),
      lambda x: sin(4 * x))
    _(Grid_1D.chebyshev(10),
      lambda x: x * (x - 1))
    _(Grid_1D.chebold(10),
      lambda x: x * (x - 1))
    _(Grid_1D.chebnew(10),
      lambda x: x * (x - 1))


def test_wedge():
    α = lambda x: sin(4 * x)
    β = lambda x: cos(4 * x)
    γ = lambda x: sin(4 * x) * cos(4 * x)

    g = Grid_1D.periodic(13)

    a0 = g.P(0, α)
    b0 = g.P(0, β)
    c0 = g.P(0, γ)
    assert a0 ^ b0 == c0

    a0 = g.P(0, α)
    b1 = g.P(1, β)
    c1 = g.P(1, γ)
    assert a0 ^ b1 == c1


def test_wedge_chebyshev():
    α = lambda x: x ** 2
    β = lambda x: (x + 1 / 2) ** 3
    γ = lambda x: x ** 2 * (x + 1 / 2) ** 3

    def _(g):
        a0 = g.P(0, α)
        b0 = g.P(0, β)
        c0 = g.P(0, γ)
        assert a0 ^ b0 == c0

        a0 = g.P(0, α)
        b1 = g.P(1, β)
        c1 = g.P(1, γ)
        assert a0 ^ b1 == c1

    _(Grid_1D.chebyshev(13))
    _(Grid_1D.chebnew(13))


def test_refine_invariant():
    np.random.seed(23)

    def _(f, h):
        g = f.grid
        gg = g.refine()
        fh = gg.P(0, lambda x: f.R(x) * h.R(x))

        xx = np.linspace(g.xmin, g.xmax, 20)
        eq(
            fh.R(xx),
            f.R(xx) * h.R(xx),
        )

    for name in [
        'periodic',
        'chebyshev',
        'chebold',
        'chebnew',
    ]:
        for N in range(2, 6):
            if name == 'periodic' and N % 2 == 0:
                continue
            g = getattr(Grid_1D, name)(N)
            G = g.dual

            # wedges
            _(g.rand(0), g.rand(0))
            _(g.rand(0), G.rand(0))
            _(G.rand(0), g.rand(0))
            _(G.rand(0), G.rand(0))

            _(g.rand(0), g.rand(1))
            _(g.rand(0), G.rand(1))
            _(G.rand(0), g.rand(1))
            _(G.rand(0), G.rand(1))

            # contraction
            _(g.rand(1), g.rand(1))
            _(g.rand(1), G.rand(1))
            _(G.rand(1), g.rand(1))
            _(G.rand(1), G.rand(1))


def test_leibniz():
    def _(g):
        D = lambda f: f.D + f.grid.BC(f.degree, f.R)

        a0 = g.rand(0)
        b0 = g.rand(0)
        assert D(a0 ^ b0) == (D(a0) ^ b0) + (a0 ^ D(b0))

        a0 = g.dual.rand(0)
        b0 = g.rand(0)
        assert D(a0 ^ b0) == (D(a0) ^ b0) + (a0 ^ D(b0))

    for N in range(1, 9):
        _(Grid_1D.periodic(N))
        _(Grid_1D.periodic(N).dual)

        # if there are more edges than vertices, this does not work
        _(Grid_1D.chebyshev(N))
        _(Grid_1D.chebold(N))

        _(Grid_1D.chebnew(N).dual)


def test_associativity_exact():
    """
    Associativity NOT satisfied by discrete forms.
    """

    N = 5
    g = Grid_1D.periodic(N)

    a0 = g.rand(0)
    b0 = g.rand(0)
    c1 = g.rand(0).D

    eq1 = a0 ^ (b0 ^ c1)
    eq2 = (a0 ^ b0) ^ c1

    assert not eq1 == eq2
