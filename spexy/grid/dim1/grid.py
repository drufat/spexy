# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
"""
>>> g = Grid_1D.chebyshev(2)
>>> g
Grid_1D.chebyshev(2)
>>> g.N, g.dual.N
((3, 2), (2, 3))
>>> g.verts()
(array([-1., -0.,  1.]),)
>>> g.edges()
(array([-1., -0.]), array([-0.,  1.]))
>>> g.dual.verts()
(array([-0.707,  0.707]),)
>>> g.dual.edges()
(array([-1.   , -0.707,  0.707]), array([-0.707,  0.707,  1.   ]))
>>> N, Nd = g.N, g.dual.N
>>> C, Cd = g.cells, g.dual.cells
>>> B, Bd = g.B, g.dual.B
>>> B[0](0)(C[0](np.arange(N[0])))
array([[ 1.,  0.,  0.]])
>>> B[0](1)(C[0](np.arange(N[0])))
array([[ 0.,  1., -0.]])
>>> B[1](0)(C[0](np.arange(N[0])))
array([[ 1.5,  0.5, -0.5]])
>>> Bd[0](0)(Cd[0](np.arange(Nd[0])))
array([[ 1.,  0.]])
>>> g.dual.dec.BC(1)(lambda x: x)
array([ 1.,  0.,  1.])
>>> g.dec.P(1)(lambda x: 1)
array([ 1.,  1.])
>>> g.delta[1](np.arange(N[1]))
array([ 1.,  1.])
>>> g.dual.dec.P(1)(lambda x: 1)
array([ 0.293,  1.414,  0.293])
>>> g.dual.delta[1](np.arange(Nd[1]))
array([ 0.293,  1.414,  0.293])
>>> f = g.dec.P(0)(lambda x: 1.)
>>> f
array([ 1.,  1.,  1.])
>>> Tf = g.dec.Pup(0)(f)
>>> Tf
array([ 1.,  1.,  1.,  1.,  1.])
>>> g.dec.Pdown(0)(Tf)
array([ 1.,  1.,  1.])
>>> g.dual.dec.Pdown(0)(Tf)
array([ 1.,  1.])

>>> g = Grid_1D.chebyshev(2)
>>> N, Nd = g.N, g.dual.N
>>> C, Cd = g.cells, g.dual.cells
>>> P, Pd = g.dec.P, g.dual.dec.P
>>> P(0)(lambda x: 1)
array([ 1.,  1.,  1.])
>>> P(0)(lambda x: x)
array([-1., -0.,  1.])
>>> P(1)(lambda x: 1)
array([ 1.,  1.])
>>> P(1)(lambda x: x)
array([-0.5,  0.5])
>>> BC, BCd = g.dec.BC, g.dual.dec.BC
>>> BCd(1)(lambda x: 1)
array([-1.,  0.,  1.])
>>> BCd(1)(lambda x: x)
array([ 1.,  0.,  1.])
>>> G, Gd = g.dec.G, g.dual.dec.G
>>> G(0)(P(0)(lambda x: x))
array([ 1.,  1.,  1.])
>>> P, Pd = g.dec.Ps, g.dual.dec.Ps
>>> P(0)(lambda x: 1)
array([ 1.,  1.,  1.])
>>> P(0)(lambda x: x)
array([-1., -0.,  1.])
>>> P(1)(lambda x: 1)
array([ 1.,  1.])
>>> P(1)(lambda x: x)
array([-0.5,  0.5])
>>> BC, BCd = g.dec.BCs, g.dual.dec.BCs
>>> BCd(1)(lambda x: 1)
array([-1.,  0.,  1.])
>>> BCd(1)(lambda x: x)
array([ 1.,  0.,  1.])
>>> g = Grid_1D.chebnew(3)
>>> g
Grid_1D.chebnew(3)
>>> g.verts()
(array([-0.5,  0.5]),)
>>> g.dual.verts()
(array([-0.866, -0.   ,  0.866]),)
>>> gg = g.refine()
>>> gg
Grid_1D.chebnew(6)
>>> gg.verts()
(array([-0.866, -0.5  , -0.   ,  0.5  ,  0.866]),)
>>> gg.refine().refine().refine()
Grid_1D.chebnew(48)
>>> gg.refine_inv()
Grid_1D.chebnew(3)
>>> gg.bndry
(-1, 1)
>>> gg.dual.bndry
"""
import numpy as np

from spexy.grid.grid import (Grid, reconstruction, bases, grids)
from spexy.integrate import (numeric, symbolic)

# numeric and symbolic integrators
Inum = numeric.integration_1d()
Isym = symbolic.integration_1d()
dim = 1


class Grid_1D(Grid):
    """
    >>> g = Grid_1D.chebyshev(3)
    >>> g.N[0]
    4
    >>> g.N[1]
    3
    >>> g.dual.N[0]
    3
    >>> g.dual.N[1]
    4
    """

    def __init__(self, name, n, dual=0):
        super().__init__()
        self.name = name
        self.n = n
        self._dual = dual

        if dual == 0:
            self.dual = Grid_1D(name, n, dual=1)
            self.dual.dual = self

    def __eq__(self, other):
        return (
            self.name == other.name and
            self.n == other.n and
            self._dual == other._dual
        )

    def __repr__(self):
        r = '{cls}.{name}({n})'.format(cls=type(self).__name__, name=self.name, n=self.n)
        if self._dual == 1:
            r = '{repr}.dual'.format(repr=r)
        return r

    @classmethod
    def make(cls, name, N):
        return cls(name, N)

    @property
    def dimension(self):
        return dim

    @property
    def N(self):
        return bases[self.name].BasesImp(self.n).numbers()[self._dual]

    @property
    def cells(self):
        return bases[self.name].BasesImp(self.n).cells()[self._dual]

    @property
    def delta(self):
        return bases[self.name].BasesImp(self.n).delta()[self._dual]

    @property
    def B(self):
        return bases[self.name].BasesImp(self.n).bases()[self._dual]

    @property
    def bndry(self):
        return bases[self.name].BasesImp(self.n).boundary()[self._dual]

    def reconst(self, d):
        return reconstruction(self.B[d], self.N[d])

    def proj_num(self, d):
        return toarray(proj(Inum[d], self.cells[d]), self.N[d])

    def proj_sym(self, d):
        return toarray(proj(Isym[d], self.cells[d]), self.N[d])

    def bndry_cond_num(self, d):
        return bndry_cond(self.bndry, self.N[d])

    def bndry_cond_sym(self, d):
        return bndry_cond(self.bndry, self.N[d])

    def deriv(self, d):
        return grids[self.name].derivative[self._dual][d]

    def hodge(self, d):
        return grids[self.name].hodge_star[self._dual][d]

    def upsample(self, d):
        return grids[self.name].upsample[self._dual][d]

    def downsample(self, d):
        return grids[self.name].downsample[self._dual][d]

    def grad(self, d):
        return grids[self.name].gradient[self._dual][d]

    @property
    def xmin(self):
        return grids[self.name].xmin

    @property
    def xmax(self):
        return grids[self.name].xmax

    def refine(self):
        return Grid_1D(self.name, 2 * self.n)

    def refine_inv(self):
        assert self.n % 2 == 0
        return Grid_1D(self.name, self.n // 2)

    @classmethod
    def enum(cls, cells, numbers):
        """
        >>> c = lambda i: i
        >>> Grid_1D.enum(c, 5)
        array([0, 1, 2, 3, 4])
        >>> c = lambda i: (i, i+1)
        >>> Grid_1D.enum(c, 5)
        (array([0, 1, 2, 3, 4]), array([1, 2, 3, 4, 5]))
        """

        return cells(np.arange(numbers))

    @property
    def deltas(self):
        return np.fromfunction(self.delta[1], (self.N[1],))


def toarray(P, N):
    """
    >>> P = lambda f: lambda i: f(i)
    >>> toarray(P, 5)(lambda i: i)
    array([ 0.,  1.,  2.,  3.,  4.])
    """
    # return lambda f: np.array(tuple(P(f)(i) for i in range(N)))
    # return lambda f: np.fromfunction(np.vectorize(P(f)), (N,))
    return lambda f: np.fromiter((P(f)(i) for i in range(N)), float, N)


def proj(I, C):
    def _(f):
        integ = I(f)
        return lambda i: integ(*C(i))

    return _


def bndry_cond(boundary, N):
    """
    >>> g = Grid_1D.regular(2)
    >>> g.dual.BC(0, lambda x: x)
    Form(1, Grid_1D.regular(2).dual, array([ 0.   ,  0.   ,  3.142]))
    >>> g.P(0, lambda x: x)
    Form(0, Grid_1D.regular(2), array([ 0.   ,  1.571,  3.142]))
    """

    def _(f):
        bc = np.zeros([N], dtype=np.float)
        if boundary:
            xmin, xmax = boundary
            bc[0] = -f(xmin)
            bc[-1] = +f(xmax)
        return bc

    return _


def bndry_cond_(boundary, N):
    """
    >>> g = Grid_1D.chebyshev(2)
    >>> bc = bndry_cond_(g.dual.bndry, g.dual.N[1])
    >>> bc(lambda x: x)
    array([ 1.,  0.,  1.])
    """

    def _(f):
        if boundary:
            xmin, xmax = boundary

            def _0(i):
                if i == 0:
                    return -f(xmin)
                if i == (N - 1):
                    return +f(xmax)
                return 0
        else:
            def _0(i):
                return 0
        return _0

    return toarray(_, N)
