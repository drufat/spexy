# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from collections import namedtuple

import numpy as np
import sympy as sy

from spexy.form import sym, coch
from spexy.grid import Grid_1D
from spexy.helper import to_matrix
from spexy.sim.save import save

Const = namedtuple('Const', ['size'])
Var = namedtuple('Var', ['L'])

@save(Const, Var)
def sim():
    size = [int(x) for x in np.exp(np.linspace(np.log(3), np.log(50), 30))]

    yield Const(size=size)

    x, y = sy.symbols('x, y')
    f = lambda x: sy.exp(sy.sin(x))
    f0 = sym.Form(0, [x], [f(x)])
    q0 = f0.D.H.D.H

    L = [[], []]
    for N in size:
        g = Grid_1D.periodic(N)

        #######################

        Lap = to_matrix(
            lambda a: coch.Form(0, g, a).D.H.D.H.array,
            g.N[0]
        )
        one = np.zeros(Lap.shape[1])
        one[0] = 1
        Lap = np.vstack((Lap, one))
        sln = np.linalg.pinv(Lap) @ np.concatenate([
            g.P(q0).array, [g.P(f0).array[0]]
        ])

        z = g.P(f0).array - sln
        err = np.linalg.norm(z, np.inf)
        L[0].append(err)

        #######################

        Lap = to_matrix(
            lambda a: coch.Form(0, g.dual, a).D.H.D.H.array,
            g.dual.N[0]
        )
        one = np.zeros(Lap.shape[1])
        one[0] = 1
        Lap = np.vstack((Lap, one))
        sln = np.linalg.pinv(Lap) @ np.concatenate([
            g.dual.P(q0).array, [g.dual.P(f0).array[0]]
        ])
        z = g.dual.P(f0).array - sln
        err = np.linalg.norm(z, np.inf)
        L[1].append(err)

        #######################

    yield Var(L=L)


