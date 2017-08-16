# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import sympy as sy

from spexy.grid.dim2.grid import Grid_2D
from spexy.sim.vorticity.plots.vort1 import gauss


def f(x, y):
    return (
        - gauss(x + sy.Rational(1, 3), y + sy.Rational(1, 3))
        + gauss(x - sy.Rational(1, 3), y + sy.Rational(1, 3))
        - gauss(x - sy.Rational(1, 3), y - sy.Rational(1, 3))
        + gauss(x + sy.Rational(1, 3), y - sy.Rational(1, 3))
    )


N = 32
g = Grid_2D.chebnew(N, N)

if __name__ == '__main__':
    from spexy.sim.vorticity.vorticity import vort_sim, vort_vv, vort_ww

    sim = vort_sim(g, f)
    sim.write('/tmp/vort.sim', 1)
    vort_vv('/tmp/vort.sim', '/tmp/vort_vv.png')
    vort_ww('/tmp/vort.sim', '/tmp/vort_ww.png')
