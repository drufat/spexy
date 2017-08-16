# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import sympy as sy

from spexy.grid.dim2.grid import Grid_2D
from spexy.sim.vorticity.plots.vort1 import gauss


def f(x, y):
    return (
        + gauss(x, y + sy.Rational(1, 3))
        + gauss(x, y - sy.Rational(1, 3))
    )


N = 32
g = Grid_2D.chebnew(N, N)
