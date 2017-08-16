# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np
import pytest
from spexy.grid import Grid_1D
from spexy.grid.grid import (
    derivative_matrix, hodge_star_matrix,
    switch_matrix, qswitch_matrix,
    upsample_matrix, downsample_matrix,
    gradient_matrix,
)
from spexy.types import periodic, chebyshev, chebnew
from spexy.ops.num import mat

gridtypes = pytest.mark.parametrize('m,Grid', [
    (periodic, Grid_1D.periodic),
    (chebyshev, Grid_1D.chebyshev),
    (chebnew, Grid_1D.chebnew),
])


@gridtypes
def test_D(m, Grid):
    for N in range(2, 7):
        g = Grid(N)
        n0, n1, n0d, n1d = g.numbers()
        d0, d0d = derivative_matrix(g)
        if not g.bndry:
            assert np.allclose(d0, mat(m.D0, n0))
        else:
            assert np.allclose(d0[1:-1, :], mat(m.D0, n0)[1:-1, :])
        if not g.dual.bndry:
            assert np.allclose(d0d, mat(m.D0d, n0d))
        else:
            assert np.allclose(d0d[1:-1, :], mat(m.D0d, n0d)[1:-1, :])


@gridtypes
def test_H(m, Grid):
    for N in range(2, 7):
        g = Grid(N)
        n0, n1, n0d, n1d = g.numbers()
        h0, h1, h0d, h1d = hodge_star_matrix(g)
        assert np.allclose(h0, mat(m.H0, n0))
        assert np.allclose(h1, mat(m.H1, n1))
        assert np.allclose(h0d, mat(m.H0d, n0d))
        assert np.allclose(h1d, mat(m.H1d, n1d))


@gridtypes
def test_S(m, Grid):
    for N in range(2, 7):
        g = Grid(N)
        n0, n1, n0d, n1d = g.numbers()
        s, sinv = switch_matrix(g)
        assert np.allclose(s, mat(m.S, n0))
        assert np.allclose(sinv, mat(m.Sinv, n0d))


@gridtypes
def test_Q(m, Grid):
    for N in range(2, 7):
        g = Grid(N)
        n0, n1, n0d, n1d = g.numbers()
        q, qinv = qswitch_matrix(g)
        assert np.allclose(q, mat(m.Q, n0))
        assert np.allclose(qinv, mat(m.Qinv, n1))


@gridtypes
def test_upsample(m, Grid):
    for N in range(2, 7):
        g, gg = Grid(N), Grid(2 * N)
        n0, n1, n0d, n1d = g.numbers()
        t0, t1, t0d, t1d = upsample_matrix(g, gg)
        assert np.allclose(t0, mat(m.Pup0, n0))
        assert np.allclose(t1, mat(m.Pup1, n1))
        assert np.allclose(t0d, mat(m.Pup0d, n0d))
        assert np.allclose(t1d, mat(m.Pup1d, n1d))


@gridtypes
def test_downsample(m, Grid):
    for N in range(2, 7):
        g, gg = Grid(N), Grid(2 * N)
        nn = gg.N[0]
        u0, u1, u0d, u1d = downsample_matrix(gg, g)
        assert np.allclose(u0, mat(m.Pdown0, nn))
        assert np.allclose(u1, mat(m.Pdown1, nn))
        assert np.allclose(u0d, mat(m.Pdown0d, nn))
        assert np.allclose(u1d, mat(m.Pdown1d, nn))


@gridtypes
def test_gradient(m, Grid):
    for N in range(2, 7):
        g = Grid(N)
        n0, n1, n0d, n1d = g.numbers()
        g0, g0d = gradient_matrix(g)
        assert np.allclose(g0, mat(m.G0, n0))
        assert np.allclose(g0d, mat(m.G0d, n0d))


@gridtypes
def test_switch(m, Grid):
    for N in range(2, 7):
        f = np.random.rand(N)
        if m in [chebyshev]:
            assert np.allclose(f, m.S(m.Sinv(f)))
        if m in [chebnew]:
            assert np.allclose(f, m.Sinv(m.S(f)))
        if m in [periodic] and (N % 2 == 1):
            assert np.allclose(f, m.Sinv(m.S(f)))
            assert np.allclose(f, m.S(m.Sinv(f)))


@gridtypes
def test_refine(m, Grid):
    for N in range(2, 7):
        f = np.random.rand(N)
        assert np.allclose(f, m.Pdown0(m.Pup0(f)))
        assert np.allclose(f, m.Pdown1(m.Pup1(f)))
        assert np.allclose(f, m.Pdown0d(m.Pup0d(f)))
        assert np.allclose(f, m.Pdown1d(m.Pup1d(f)))
