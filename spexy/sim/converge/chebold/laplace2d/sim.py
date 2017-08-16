# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from collections import namedtuple

import numpy as np
import sympy as sy

from spexy.form import sym
from spexy.grid import Grid_2D
from spexy.sim.save import save

Const = namedtuple('Const', ['size'])
Var = namedtuple('Var', ['L'])


@save(Const, Var)
def sim():
    size = [int(x) for x in np.exp(np.linspace(np.log(3), np.log(30), 30))]

    yield Const(size=size)

    x, y = sy.symbols('x, y')
    f0 = sym.Form(0, [x, y], [sy.exp(x) + sy.exp(y)])
    q0 = f0.D.H.D.H

    f1 = sym.Form(1, [x, y], [sy.exp(x), sy.exp(y)])
    q1 = f1.D.H.D.H + f1.H.D.H.D

    L = [[], [], [], []]
    for N in size:
        g = Grid_2D.chebyshev(N, N)

        bc = g.dual.BC(f0.D.H)
        z = (g.P(f0).D.H.D + bc).H - g.P(q0)
        L[0].append(
            np.linalg.norm(z.array, np.inf)
        )

        bc = g.dual.BC(f0)
        z = (g.dual.P(f0).D + bc).H.D.H - g.dual.P(q0)
        L[1].append(
            np.linalg.norm(z.array, np.inf)
        )

        # normal, and curl
        f = g.P(f1)
        bc1 = g.dual.BC(f1.H)
        bc0 = g.dual.BC(f1.D.H)
        z = ((f.H.D + bc1).H.D + (f.D.H.D + bc0).H) - g.P(q1)
        L[2].append(
            np.linalg.norm(z.array, np.inf)
        )

        # tangent, and divergence
        f = g.dual.P(f1)
        bc1 = g.dual.BC(f1)
        bc0 = g.dual.BC(f1.H.D.H)
        z = ((f.H.D.H.D + bc0) + (f.D + bc1).H.D.H) - g.dual.P(q1)
        L[3].append(
            np.linalg.norm(z.array, np.inf)
        )

    yield Var(L=L)


