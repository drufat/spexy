# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import sympy as sy

from spexy.grid.dim2.grid import Grid_2D

σ = sy.Rational(1, 10)


def gauss(x, y):
    return sy.exp(-(x ** 2 + y ** 2) / σ ** 2)


def f(x, y):
    return (
        - gauss(x, y + sy.Rational(1, 3))
        + gauss(x, y - sy.Rational(1, 3))
    )


N = 32
g = Grid_2D.chebnew(N, N)

if __name__ == '__main__':
    from spexy.sim.vorticity.vorticity import vort_sim, vort_vv, vort_ww

    sim = vort_sim(g, f)
    sim.write('/tmp/vort.sim', 1)
    vort_vv('/tmp/vort.sim', '/tmp/vort_vv.png')
    vort_ww('/tmp/vort.sim', '/tmp/vort_ww.png')
