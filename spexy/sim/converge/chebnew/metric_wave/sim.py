# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from collections import namedtuple

import numpy as np
import sympy as sy

from spexy.embed.jacob import SJ
from spexy.embed.surface.wave import s
from spexy.form import coch, sym
from spexy.form.sym import pullbackform
from spexy.grid import Grid_2D
from spexy.sim.converge.chebnew.metric.sim import f0, f1
from spexy.sim.converge.loglog import loglogint
from spexy.sim.metric import bases_sqrt
from spexy.sim.save import save

x, y = sy.symbols('x, y')
vx, vy = sy.symbols('vx, vy')
α = sym.Form(0, [x, y], [f0(x, y)])
β = α.D.H.D.H
αDH = α.D.H

M = lambda u, v: s(u, v, 0)[:2]
pull = pullbackform(M)
αpull = pull(α)
βpull = pull(β)
αDHpull = pull(αDH)

α1 = sym.Form(1, [x, y], f1(x, y))
β1 = α1.D.H.D.H + α1.H.D.H.D

α1pull = pull(α1)
β1pull = pull(β1)
α1Hpull = pull(α1.H)
α1DHpull = pull(α1.D.H)
α1HDHpull = pull(α1.H.D.H)

Const = namedtuple('Const', ['size'])
Var = namedtuple('Var', ['L0', 'L0d', 'L1', 'L1d'])


@save(Const, Var)
def sim():
    size = loglogint(4, 80, 20)

    yield Const(size=size)

    L0, L0d, L1, L1d = [], [], [], []
    for N in size:
        g = Grid_2D.chebnew(N, N)

        # Orthonormal frame J, Jinv
        S0, J0 = SJ(s, g.blk)(0)
        J, Jinv = bases_sqrt(*J0)
        J = [coch.Form(0, g.blk, [_]).Pup.comp[0].array for _ in J]
        Jinv = [coch.Form(0, g.blk, [_]).Pup.comp[0].array for _ in Jinv]

        f = αpull.P(g)
        href = βpull.P(g)
        bc = g.BC(αpull)
        h = (f.blk.D + bc.blk).HH(J, Jinv).D.HH(J, Jinv).blkinv
        z = h - href
        err = np.linalg.norm(z.array, np.inf)
        L0.append(err)

        f = αpull.P(g.dual)
        href = βpull.P(g.dual)
        bc = g.BC(αDHpull)
        h = (f.blk.D.HH(J, Jinv).D + bc.blk).HH(J, Jinv).blkinv
        z = h - href
        err = np.linalg.norm(z.array, np.inf)
        L0d.append(err)

        f = α1pull.P(g)
        href = β1pull.P(g)
        bc0 = g.BC(α1HDHpull)
        bc1 = g.BC(α1pull)
        h = (
            (f.blk.D + bc1.blk).HH(J, Jinv).D.HH(J, Jinv).blkinv +
            (f.blk.HH(J, Jinv).D.HH(J, Jinv).D + bc0.blk).blkinv
        )
        z = h - href
        err = np.linalg.norm(z.array, np.inf)
        L1.append(err)

        f = α1pull.P(g.dual)
        href = β1pull.P(g.dual)
        bc0 = g.BC(α1DHpull)
        bc1 = g.BC(α1Hpull)
        h = (
            (f.blk.D.HH(J, Jinv).D + bc0.blk).HH(J, Jinv).blkinv +
            (f.blk.HH(J, Jinv).D + bc1.blk).HH(J, Jinv).D.blkinv
        )
        z = h - href
        err = np.linalg.norm(z.array, np.inf)
        L1d.append(err)

        print(N)

    yield Var(L0=L0, L0d=L0d, L1=L1, L1d=L1d)


if __name__ == '__main__':
    sim()
