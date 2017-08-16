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

    # f = lambda x: x**2
    # f = lambda x: x**3
    f = lambda x: sy.exp(x)

    f0 = sym.Form(0, [x], [f(x)])
    q0 = f0.D.H.D.H

    L = [[], []]
    for N in size:
        g = Grid_1D.chebyshev(N)

        #######################

        # Dirichlet boundary conditions
        # Solve: H1(D(H1d(DD(f) + bc))) == q

        Lap = to_matrix(
            lambda a: coch.Form(0, g.dual, a).D.H.D.H.array,
            g.dual.N[0]
        )
        bc = g.dual.BC(f0)

        z = np.linalg.inv(Lap) @ (g.dual.P(q0) - bc.H.D.H).array
        err = np.linalg.norm(g.dual.P(f0).array - z, np.inf)
        L[0].append(err)

        #######################

        # Neumann boundary conditions
        # Solve: H1d(DD(  H1(D0(x))) + bc) == q

        Lap = to_matrix(
            lambda a: coch.Form(0, g, a).D.H.D.H.array,
            g.N[0]
        )
        bc = g.dual.BC(f0.D.H)

        # The matrix is not invertible, because it is defined up to a constant,
        # first derivative. Use linalg.pinv instead of linalg.inv
        one = np.zeros(Lap.shape[1])
        one[0] = 1
        Lap = np.vstack((Lap, one))
        Lapinv = np.linalg.pinv(Lap)

        z = Lapinv @ np.concatenate([(g.P(q0) - bc.H).array, [g.P(f0).array[0]]])
        err = np.linalg.norm(g.P(f0).array - z, np.inf)

        L[1].append(err)

        #######################

    yield Var(L=L)
