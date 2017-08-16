# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np
from numpy.testing import assert_array_almost_equal as eq

import spexy.integrate.integrate as integrate
from spexy.grid import Grid_1D


def test_integrals():
    for N in (6, 7, 8, 9):

        g = Grid_1D.periodic(N)
        pnts = np.concatenate([g.verts()[0], [g.xmax]])
        p0, p1 = pnts[:-1], pnts[1:]
        for f in (np.sin,
                  np.cos):
            reference = integrate.integrate_quad(f, p0, p1)
            eq(integrate.integrate_boole1(pnts, f), reference)
            eq(integrate.integrate_spectral_coarse(pnts, f), reference)
            eq(integrate.integrate_spectral(pnts, f), reference)

        g = Grid_1D.chebyshev(N)
        pnts = g.verts()[0]
        p0, p1 = pnts[:-1], pnts[1:]
        for f in ((lambda x: x),
                  (lambda x: x ** 3),
                  np.exp):
            reference = integrate.integrate_quad(f, p0, p1)
            eq(integrate.integrate_boole1(pnts, f), reference)
            eq(integrate.integrate_chebyshev(pnts, f), reference)

        g = Grid_1D.chebyshev(N)
        pnts = np.concatenate(([-1], g.dual.verts()[0], [+1]))
        p0, p1 = pnts[:-1], pnts[1:]
        for f in ((lambda x: x),
                  (lambda x: x ** 3),
                  np.exp):
            reference = integrate.integrate_quad(f, p0, p1)
            eq(integrate.integrate_boole1(pnts, f), reference)
            eq(integrate.integrate_chebyshev_dual(pnts, f), reference)
