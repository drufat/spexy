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
    size = [int(x) for x in np.exp(np.linspace(np.log(3), np.log(50), 30))]

    yield Const(size=size)

    a = lambda x: sy.exp(sy.sin(x))

    x, y = sy.symbols('x, y')
    f0 = sym.Form(0, [x, y], [a(x) + a(y)])
    q0 = f0.D.H.D.H

    f1 = sym.Form(1, [x, y], [a(x), a(y)])
    q1 = f1.D.H.D.H + f1.H.D.H.D

    L = [[], [], [], []]
    for N in size:
        g = Grid_2D.periodic(N, N)

        z = g.P(f0).D.H.D.H - g.P(q0)
        L[0].append(
            np.linalg.norm(z.array, np.inf)
        )

        z = g.dual.P(f0).D.H.D.H - g.dual.P(q0)
        L[1].append(
            np.linalg.norm(z.array, np.inf)
        )

        f = g.P(f1)
        z = (f.D.H.D.H + f.H.D.H.D) - g.P(q1)
        L[2].append(
            np.linalg.norm(z.array, np.inf)
        )

        f = g.dual.P(f1)
        z = (f.D.H.D.H + f.H.D.H.D) - g.dual.P(q1)
        L[3].append(
            np.linalg.norm(z.array, np.inf)
        )

    yield Var(L=L)


