# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.

import numpy as np

from spexy.grid.dim2.grid import Grid_2D
from spexy.sim.vorticity.plots.vort1 import gauss

np.random.seed(7)
r = np.random.rand(36, 3)


def f(x, y):
    rslt = 0
    for x0, y0, z0 in r:
        s0 = np.sign(2 * z0 - 1)
        x0 = 2 * x0 - 1
        y0 = 2 * y0 - 1
        rslt += s0 * gauss(x - x0, y - y0)
    return rslt


N = 32
g = Grid_2D.chebnew(N, N)
