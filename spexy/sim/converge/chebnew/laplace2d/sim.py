# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from collections import namedtuple

import numpy as np
import sympy as sy

from spexy.form import sym
from spexy.grid import Grid_2D
from spexy.sim.converge.loglog import loglogint
from spexy.sim.save import save

Const = namedtuple('Const', ['size'])
Var = namedtuple('Var', ['L0', 'L0d', 'L1', 'L1d'])


@save(Const, Var)
def sim():
    size = loglogint(4, 24, 20)

    yield Const(size=size)

    x, y = sy.symbols('x, y')
    f0 = sym.Form(0, [x, y], [sy.exp(x) + sy.exp(y)])
    q0 = f0.D.H.D.H

    f1 = sym.Form(1, [x, y], [sy.exp(x), sy.exp(y)])
    q1 = f1.D.H.D.H + f1.H.D.H.D

    L0, L0d, L1, L1d = [], [], [], []
    for N in size:
        g = Grid_2D.chebnew(N, N)

        bc = g.BC(f0)
        z = (g.P(f0).D + bc).H.D.H - g.P(q0)
        L0.append(
            np.linalg.norm(z.array, np.inf)
        )

        bc = g.BC(f0.D.H)
        z = (g.dual.P(f0).D.H.D + bc).H - g.dual.P(q0)
        L0d.append(
            np.linalg.norm(z.array, np.inf)
        )

        # tangent, and divergence
        f = g.P(f1)
        bc1 = g.BC(f1)
        bc0 = g.BC(f1.H.D.H)
        z = ((f.H.D.H.D + bc0) + (f.D + bc1).H.D.H) - g.P(q1)
        L1.append(
            np.linalg.norm(z.array, np.inf)
        )

        # normal, and curl
        f = g.dual.P(f1)
        bc1 = g.BC(f1.H)
        bc0 = g.BC(f1.D.H)
        z = ((f.H.D + bc1).H.D + (f.D.H.D + bc0).H) - g.dual.P(q1)
        L1d.append(
            np.linalg.norm(z.array, np.inf)
        )

    yield Var(L0=L0, L0d=L0d, L1=L1, L1d=L1d)
