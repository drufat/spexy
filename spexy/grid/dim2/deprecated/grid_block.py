# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np

from spexy.integrate import (numeric, symbolic)
from spexy.grid.dim2.grid_block import (toarray, bndry_cond, eqv, GridBlock_2D)
from spexy.strides import gen, forward


def toarray_(P, shape):
    """
    >>> shape = ((3, 4), (5, 6))
    >>> P = lambda f: (lambda i, j: 100*i + j, lambda i, j: i + 10*j)
    >>> A = toarray(P(None), shape)
    >>> B = toarray_(P, shape)(None)
    >>> assert eqv(A, B)
    """

    def _(f):
        itr = (P(f)[k](i, j) for k, i, j in gen(shape))
        return np.array(forward(shape)(np.fromiter(itr, dtype=np.float)))

    return _


def bndry_cond_(I, C, N):
    """
    >>> I = numeric.integration_2d_regular()
    >>> Is = symbolic.integration_2d_regular()
    >>> g = GridBlock_2D.chebyshev(2, 5)
    >>> bc = bndry_cond(g.dual.gx.bndry, g.dual.gy.bndry, g.dual.gx.proj_num, g.dual.gy.proj_num, g.dual.N)
    >>> bc_ = bndry_cond_(I, g.dual.cells, g.dual.N)

    >>> f = lambda x, y: (np.exp(-x) * y,)
    >>> assert eqv(bc[1](f), bc_[1](f))

    >>> f = lambda x, y: (-y, x * y)
    >>> assert eqv(bc[2](f), bc_[2](f))

    >>> f = lambda x, y: (x*y, )
    >>> assert eqv(bc[1](f), bc_[1](f))
    """

    def BC1(f):
        (Ni, o), (o, Nj) = N[1]
        Ci, Cj = C[1]
        p = I[0](f)

        def _0(i, j):
            b = 0.0
            if i == 0:
                x0, x1, y = Ci(i, j)
                b -= p[0](x0, y)
            elif i == Ni - 1:
                x0, x1, y = Ci(i, j)
                b += p[0](x1, y)
            return b

        def _1(i, j):
            b = 0.0
            if j == 0:
                x, y0, y1 = Cj(i, j)
                b -= p[0](x, y0)
            elif j == Nj - 1:
                x, y0, y1 = Cj(i, j)
                b += p[0](x, y1)
            return b

        return (_0, _1)

    def BC2(f):
        (Ni, Nj), = N[2]
        Cij, = C[2]
        p = I[1](f)

        def _0(i, j):
            b = 0.0
            if i == 0:
                x0, x1, y0, y1 = Cij(i, j)
                b -= p[1](x0, y0, y1)
            elif i == Ni - 1:
                x0, x1, y0, y1 = Cij(i, j)
                b += p[1](x1, y0, y1)
            if j == 0:
                x0, x1, y0, y1 = Cij(i, j)
                b += p[0](x0, x1, y0)
            elif j == Nj - 1:
                x0, x1, y0, y1 = Cij(i, j)
                b -= p[0](x0, x1, y1)
            return b

        return (_0,)

    return (
        lambda f: (0,),
        lambda f: toarray(BC1(f), N[1]),
        lambda f: toarray(BC2(f), N[2]),
    )
