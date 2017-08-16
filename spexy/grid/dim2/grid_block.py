# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
"""
Spectral DEC in 2D
=============================
Representation of a cellular complex in 2D (block array):

vertices: (
    (x, y),
)
edges: (
    (x0, x1, y),
    (x, y0, y1),
)
faces: (
    (x0, x1, y0, y1),
)

------------> y
|
|
|
|
\/  x

>>> g = GridBlock_2D.chebyshev(2, 2)
>>> g
GridBlock_2D.chebyshev(2, 2)
>>> g.dual
GridBlock_2D.chebyshev(2, 2).dual
>>> g.N[1]
((2, 3), (3, 2))
>>> g.dual.N[1]
((3, 2), (2, 3))
>>> (x, y), = g.verts()
>>> x
array([[-1., -1., -1.],
       [-0., -0., -0.],
       [ 1.,  1.,  1.]])
>>> y
array([[-1., -0.,  1.],
       [-1., -0.,  1.],
       [-1., -0.,  1.]])
>>> x.flags['C_CONTIGUOUS']
True
>>> (x0, x1, y),(x, y0, y1) = g.edges()
>>> x0
array([[-1., -1., -1.],
       [-0., -0., -0.]])
>>> y
array([[-1., -0.,  1.],
       [-1., -0.,  1.]])
>>> (x0, x1, y0, y1), = g.faces()
>>> x0
array([[-1., -1.],
       [-0., -0.]])
>>> y1
array([[-0.,  1.],
       [-0.,  1.]])
>>> g.dec.P(0)(lambda x, y: (1.,))[0]
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])
>>> f = g.dec.P(0)(lambda x, y: (1.,))
>>> g.dec.Pup(0)(f)[0][0]
array([[ 1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.]])
>>> Tf = g.dec.Pup(0)(f)
>>> g.dec.Pdown(0)(Tf)[0]
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])
>>> P0, P1, P2, P0d, P1d, P2d = g.projection()
>>> P0(lambda x, y: (1,))[0]
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])
>>> P1(lambda x, y: (1, 0))[0]
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])
>>> P1(lambda x, y: (1, 0))[1]
array([[ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.]])
>>> P2(lambda x, y: (1,))[0]
array([[ 1.,  1.],
       [ 1.,  1.]])
>>> P = g.dec.Ps
>>> P(0)(lambda x, y: (1,))[0]
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])
>>> P(1)(lambda x, y: (1, 0))[0]
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])
>>> P(1)(lambda x, y: (1, 0))[1]
array([[ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.]])
>>> P(2)(lambda x, y: (1,))[0]
array([[ 1.,  1.],
       [ 1.,  1.]])
>>> G = g.grad(0)
>>> grad = G(P(0)(lambda x, y: (x,)))
>>> grad[0][0]
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])
>>> grad[1][0] + 0.0
array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.],
       [ 0.,  0.,  0.]])
>>> bc = g.dual.dec.BC
>>> bc(1)(lambda x, y: (1,))[0]
array([[-1., -1.],
       [ 0.,  0.],
       [ 1.,  1.]])
>>> bc(1)(lambda x, y: (1,))[1]
array([[-1.,  0.,  1.],
       [-1.,  0.,  1.]])
>>> g.gx.dual.dec.P(0)(lambda x: 1)
array([ 1.,  1.])
>>> bc(2)(lambda x, y: (1, 0))[0]
array([[ 0.293,  0.   , -0.293],
       [ 1.414,  0.   , -1.414],
       [ 0.293,  0.   , -0.293]])
>>> bc(2)(lambda x, y: (0, 1))[0]
array([[-0.293, -1.414, -0.293],
       [ 0.   ,  0.   ,  0.   ],
       [ 0.293,  1.414,  0.293]])
>>> bc = g.dual.dec.BCs
>>> bc(2)(lambda x, y: (1, 0))[0]
array([[ 0.293,  0.   , -0.293],
       [ 1.414,  0.   , -1.414],
       [ 0.293,  0.   , -0.293]])
>>> g.refine()
GridBlock_2D.chebyshev(4, 4)
>>> g.refine().refine_inv()
GridBlock_2D.chebyshev(2, 2)
"""
import numpy as np

from spexy.grid.dim1.grid import Grid_1D
from spexy.grid.dim1.samples import (cells_lambdify, cells_012, cells_345, )
from spexy.grid.grid import Grid
from spexy.integrate import (numeric, symbolic)
from spexy.strides import gen

# numeric and symbolic integrators
Inum = numeric.integration_2d_regular()
Isym = symbolic.integration_2d_regular()
dim = 2


class GridBlock_2D(Grid):
    def __init__(self, gx, gy, dual=0):
        super().__init__()
        assert gx.dimension == 1
        assert gy.dimension == 1
        self.gx = gx
        self.gy = gy
        self._dual = dual

        if dual == 0:
            self.dual = GridBlock_2D(gx.dual, gy.dual, dual=1)
            self.dual.dual = self

    def __eq__(self, other):
        return (
            self.gx == other.gx and
            self.gy == other.gy
        )

    def __repr__(self):
        N, M = self.gx.n, self.gy.n
        if self.gx.name == self.gy.name:
            rpr = '{cls}.{name}({N}, {M})'.format(cls=type(self).__name__, name=self.gx.name, N=N, M=M)
        else:
            rpr = '{cls}({gx}, {gy})'.format(cls=type(self).__name__, gx=self.gx, gy=self.gy)
        if self._dual == 1:
            rpr = '{repr}.dual'.format(repr=rpr)
        return rpr

    @classmethod
    def make(cls, name, N, M):
        return cls(Grid_1D(name, N), Grid_1D(name, M))

    @property
    def dimension(self):
        return dim

    @property
    def N(self):
        return mesh_numbers(self.gx.N, self.gy.N)

    @property
    def cells(self):
        return mesh_cells(self.gx.cells, self.gy.cells)

    @property
    def delta(self):
        return mesh_delta(self.gx.delta, self.gy.delta)

    @property
    def B(self):
        return mesh_basis_fn(self.gx.B, self.gy.B)

    def reconst(self, d):
        return reconstruction(self.B[d], self.N[d])

    def proj_num(self, d):
        P = proj(Inum[d], self.cells[d])
        return lambda f: toarray(P(f), self.N[d])

    def proj_sym(self, d):
        P = proj(Isym[d], self.cells[d])
        return lambda f: toarray(P(f), self.N[d])

    def bndry_cond_num(self, d):
        return bndry_cond(self.gx.bndry, self.gy.bndry, self.gx.proj_num, self.gy.proj_num, self.N)[d]

    def bndry_cond_sym(self, d):
        return bndry_cond(self.gx.bndry, self.gy.bndry, self.gx.proj_sym, self.gy.proj_sym, self.N)[d]

    def deriv(self, d):
        return derivative(self.gx.deriv, self.gy.deriv)[d]

    def hodge(self, d):
        return hodge_star(self.gx.hodge, self.gy.hodge)[d]

    def upsample(self, d):
        T = refining(self.gx.upsample, self.gy.upsample)[d]
        return lambda f: tuple((_,) for _ in T(f))

    def downsample(self, d):
        U = refining(self.gx.downsample, self.gy.downsample)[d]
        return lambda f: U(tuple(_[0] for _ in f))

    def grad(self, d):
        assert d == 0
        G = gradient(self.gx.grad, self.gy.grad)
        return lambda f: tuple((_,) for _ in G(f))

    def refine(self):
        return GridBlock_2D(self.gx.refine(), self.gy.refine())

    def refine_inv(self):
        return GridBlock_2D(self.gx.refine_inv(), self.gy.refine_inv())

    @classmethod
    def enum(cls, cells, numbers):
        return enum(cells, numbers)


along = [
    lambda h: lambda x: h(x.T).T,
    lambda h: lambda x: h(x),
]


def derivative(dx, dy):
    D0x, D1x = along[0](dx(0)), along[0](dx(1))
    D0y, D1y = along[1](dy(0)), along[1](dy(1))

    _0 = lambda f: (D0x(f[0]), D0y(f[0]),)
    _1 = lambda f: (-D0y(f[0]) + D0x(f[1]),)
    _2 = lambda f: 0

    return _0, _1, _2


def hodge_star(hx, hy):
    H0x, H1x = along[0](hx(0)), along[0](hx(1))
    H0y, H1y = along[1](hy(0)), along[1](hy(1))

    _0 = lambda f: (H0x(H0y(f[0])),)
    _1 = lambda f: (-H0x(H1y(f[1])), H0y(H1x(f[0])))
    _2 = lambda f: (H1x(H1y(f[0])),)

    return _0, _1, _2


def refining(tx, ty):
    T0x, T1x = along[0](tx(0)), along[0](tx(1))
    T0y, T1y = along[1](ty(0)), along[1](ty(1))

    _0 = lambda f: (T0x(T0y(f[0])),)
    _1 = lambda f: (T0y(T1x(f[0])), T0x(T1y(f[1])))
    _2 = lambda f: (T1x(T1y(f[0])),)

    return _0, _1, _2


def gradient(gx, gy):
    G0x, G1x = along[0](gx(0)), along[0](gx(1))
    G0y, G1y = along[1](gy(0)), along[1](gy(1))

    return lambda f: (G0x(f[0]), G0y(f[0]))


def mesh_numbers(Nx, Ny):
    """
    >>> gx = Grid_1D.chebyshev(2)
    >>> gy = Grid_1D.chebyshev(5)
    >>> S0, S1, S2 = mesh_numbers(gx.N, gy.N)
    >>> S0d, S1d, S2d = mesh_numbers(gx.dual.N, gy.dual.N)
    >>> S0
    ((3, 6),)
    >>> S1
    ((2, 6), (3, 5))
    >>> S1d
    ((3, 5), (2, 6))
    >>> S2
    ((2, 5),)
    """

    vx, ex = Nx[0], Nx[1]
    vy, ey = Ny[0], Ny[1]

    v = (
        (vx, vy),
    )
    e = (
        (ex, vy),
        (vx, ey),
    )
    f = (
        (ex, ey),
    )
    return v, e, f


def mesh_cells(cellsx, cellsy):
    """
    >>> cx, nx = cells_lambdify(*cells_012()[0])
    >>> cy, ny = cells_lambdify(*cells_345()[0])
    >>> c0, c1, c2 = mesh_cells(cx, cy)
    >>> cx, nx = cells_lambdify(*cells_012()[1])
    >>> cy, ny = cells_lambdify(*cells_345()[1])
    >>> c0d, c1d, c2d = mesh_cells(cx, cy)
    >>> c0[0](0, 0)
    (0, 3)
    >>> c0[0](1, 0)
    (1, 3)
    >>> c0[0](0, 1)
    (0, 4)
    >>> c1[0](0, 0)
    (0, 1, 3)
    >>> c1[1](0, 0)
    (0, 3, 4)
    >>> c1[1](0, 1)
    (0, 4, 5)
    >>> c2[0](0, 0)
    (0, 1, 3, 4)
    """

    vx, ex = cellsx[0], cellsx[1]
    vy, ey = cellsy[0], cellsy[1]

    v = (
        lambda i, j: (*vx(i), *vy(j)),
    )
    e = (
        lambda i, j: (*ex(i), *vy(j)),
        lambda i, j: (*vx(i), *ey(j)),
    )
    f = (
        lambda i, j: (*ex(i), *ey(j)),
    )

    return v, e, f


def mesh_delta(deltax, deltay):
    vx, ex = deltax[0], deltax[1]
    vy, ey = deltay[0], deltay[1]

    v = (
        lambda i, j: vx(i) * vy(j),
    )
    e = (
        lambda i, j: ex(i) * vy(j),
        lambda i, j: vx(i) * ey(j),
    )
    f = (
        lambda i, j: ex(i) * ey(j),
    )

    return v, e, f


def mesh_basis_fn(bx, by):
    """
    >>> gx = Grid_1D.chebyshev(2)
    >>> B = mesh_basis_fn(gx.B, gx.B)
    >>> x, y = np.linspace(-1, 1, 3), np.linspace(-1, 1, 3)
    >>> B[1][1](1, 0)(x, y)
    array([array([0]), array([ 0. ,  0.5,  0. ])], dtype=object)
    """

    vx, ex = bx[0], bx[1]
    vy, ey = by[0], by[1]

    v = (
        lambda i, j: lambda x, y: np.array([vx(i)(x) * vy(j)(y)]),
    )
    e = (
        lambda i, j: lambda x, y: np.array([ex(i)(x) * vy(j)(y), np.array([0])]),
        lambda i, j: lambda x, y: np.array([np.array([0]), vx(i)(x) * ey(j)(y)]),
    )
    f = (
        lambda i, j: lambda x, y: np.array([ex(i)(x) * ey(j)(y)]),
    )
    return v, e, f


def eqv(A, B):
    return all(np.allclose(a, b) for a, b in zip(A, B))


def meshgrid(*args):
    return np.meshgrid(*args, indexing='ij')


def enum(cells, shapes):
    """
    >>> shapes = ((3, 4),)
    >>> cells = (lambda i, j: i + 10*j,)
    >>> e = enum(cells, shapes)
    >>> e[0]
    array([[ 0, 10, 20, 30],
           [ 1, 11, 21, 31],
           [ 2, 12, 22, 32]])
    >>> shapes = ((3, 4), (5, 6))
    >>> cells =  (lambda i, j: 100*i + j, lambda i, j: i + 10*j)
    >>> e = enum(cells, shapes)
    >>> e[0]
    array([[  0,   1,   2,   3],
           [100, 101, 102, 103],
           [200, 201, 202, 203]])
    >>> e[1]
    array([[ 0, 10, 20, 30, 40, 50],
           [ 1, 11, 21, 31, 41, 51],
           [ 2, 12, 22, 32, 42, 52],
           [ 3, 13, 23, 33, 43, 53],
           [ 4, 14, 24, 34, 44, 54]])
    """

    def _(cell, shape):
        return cell(*meshgrid(*(np.arange(n) for n in shape)))

    return np.array([_(cell, shape) for cell, shape in zip(cells, shapes)])


def toarray(cells, shape):
    """
    >>> f = None
    >>> shape = ((3, 4),)
    >>> cells = (lambda i, j: 10*i + j,)
    >>> A = toarray(cells, shape)
    >>> A[0]
    array([[  0.,   1.,   2.,   3.],
           [ 10.,  11.,  12.,  13.],
           [ 20.,  21.,  22.,  23.]])
    >>> shape = ((3, 4), (5, 6))
    >>> cells = (lambda i, j: 100*i + j, lambda i, j: i + 10*j)
    >>> A = toarray(cells, shape)
    >>> A[0]
    array([[   0.,    1.,    2.,    3.],
           [ 100.,  101.,  102.,  103.],
           [ 200.,  201.,  202.,  203.]])
    >>> A[1]
    array([[  0.,  10.,  20.,  30.,  40.,  50.],
           [  1.,  11.,  21.,  31.,  41.,  51.],
           [  2.,  12.,  22.,  32.,  42.,  52.],
           [  3.,  13.,  23.,  33.,  43.,  53.],
           [  4.,  14.,  24.,  34.,  44.,  54.]])
    >>> B = cells
    >>> for k, i, j in gen(shape):
    ...     assert A[k][i, j] == B[k](i, j)
    """

    def _(cell, shape):
        cell = np.vectorize(cell)
        return cell(*meshgrid(*(np.arange(n) for n in shape))).astype(np.float)

    return np.array([_(cells[k], shape[k]) for k in range(len(shape))])


def reconstruction(base, shape):
    """
    >>> gx, gy = Grid_1D.chebyshev(2), Grid_1D.chebyshev(2)
    >>> shapes = mesh_numbers(gx.N, gy.N)
    >>> bases = mesh_basis_fn(gx.B, gy.B)
    >>> b = bases[1]
    >>> s = shapes[1]
    >>> r = reconstruction(b, s)
    >>> f = np.array([np.zeros(s[0]), np.zeros(s[1])])
    >>> f[0][0, 0] = 1
    >>> r(f)(-1, -1)
    array([array([ 1.5]), 0.0], dtype=object)
    """
    return lambda f: (
        lambda x, y: (
            sum(f[k][i, j] * base[k](i, j)(x, y) for k, i, j in gen(shape))
        )
    )


def proj(I, C):
    def P(f):
        integ = I(f)
        cell = C
        return tuple(
            (
                lambda k: lambda i, j: integ[k](*cell[k](i, j))
            )(k) for k in range(len(integ))
        )

    return P


def bndry_cond(bndx, bndy, Px, Py, N):
    """
    >>> g = GridBlock_2D.chebyshev(2, 2)
    >>> g = g.dual
    >>> _0, _1, _2 = bndry_cond(g.gx.bndry, g.gy.bndry, g.gx.proj_num, g.gy.proj_num, g.N)
    >>> _1(lambda x, y: (1,))[0]
    array([[-1., -1.],
           [ 0.,  0.],
           [ 1.,  1.]])
    >>> _1(lambda x, y: (1,))[1]
    array([[-1.,  0.,  1.],
           [-1.,  0.,  1.]])
    >>> _2(lambda x, y: (1, 0))[0]
    array([[ 0.293,  0.   , -0.293],
           [ 1.414,  0.   , -1.414],
           [ 0.293,  0.   , -0.293]])
    >>> _2(lambda x, y: (0, 1))[0]
    array([[-0.293, -1.414, -0.293],
           [ 0.   ,  0.   ,  0.   ],
           [ 0.293,  1.414,  0.293]])
    >>> _0, _1, _2 = bndry_cond(g.gx.bndry, g.gy.bndry, g.gx.proj_sym, g.gy.proj_sym, g.N)
    >>> _2(lambda x, y: (1, 0))[0]
    array([[ 0.293,  0.   , -0.293],
           [ 1.414,  0.   , -1.414],
           [ 0.293,  0.   , -0.293]])
    """

    def _0(*args):
        return 0

    def _1(*args):
        P0x, P0y = Px(0), Py(0)
        if len(args) == 1:
            (f,) = args
            fxmin = lambda _: f(xmin, _)[0]
            fxmax = lambda _: f(xmax, _)[0]
            fymin = lambda _: f(_, ymin)[0]
            fymax = lambda _: f(_, ymax)[0]
        else:
            (fxmin, fxmax, fymin, fymax) = args
        bc = np.array([np.zeros(shape, dtype=np.float) for shape in N[1]])
        if bndx:
            xmin, xmax = bndx
            bc[0][0, :] -= P0y(fxmin)
            bc[0][-1, :] += P0y(fxmax)
        if bndy:
            ymin, ymax = bndy
            bc[1][:, 0] -= P0x(fymin)
            bc[1][:, -1] += P0x(fymax)
        return bc

    def _2(*args):
        P1x, P1y = Px(1), Py(1)
        if len(args) == 1:
            (f,) = args
            fxmin = lambda _: f(xmin, _)[1]
            fxmax = lambda _: f(xmax, _)[1]
            fymin = lambda _: f(_, ymin)[0]
            fymax = lambda _: f(_, ymax)[0]
        else:
            (fxmin, fxmax, fymin, fymax) = args
        bc = np.array([np.zeros(shape, dtype=np.float) for shape in N[2]])
        if bndx:
            xmin, xmax = bndx
            bc[0][0, :] -= P1y(fxmin)
            bc[0][-1, :] += P1y(fxmax)
        if bndy:
            ymin, ymax = bndy
            bc[0][:, 0] += P1x(fymin)
            bc[0][:, -1] -= P1x(fymax)
        return bc

    return _0, _1, _2
