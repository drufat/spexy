# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
r"""
Representation of a cellular complex in 2D (flat array):

vertices: (x, y)
edges:    (x0, y0, x1, y1)
faces:    (x0, y0, x1, y1, x2, y2, x3, y3)

>>> g = Grid_2D.chebyshev(2, 2)
>>> g
Grid_2D.chebyshev(2, 2)
>>> g.numbers()
(9, 12, 4, 4, 12, 9)
>>> GridBlock_2D.chebyshev(2, 2).numbers()
(((3, 3),), ((2, 3), (3, 2)), ((2, 2),), ((2, 2),), ((3, 2), (2, 3)), ((3, 3),))
>>> x, y = g.verts()
>>> x
array([-1., -1., -1., -0., -0., -0.,  1.,  1.,  1.])
>>> y
array([-1., -0.,  1., -1., -0.,  1., -1., -0.,  1.])
>>> x0, y0, x1, y1 = g.edges()
>>> x0
array([-1., -1., -1., -0., -0., -0., -1., -1., -0., -0.,  1.,  1.])
>>> y0
array([-1., -0.,  1., -1., -0.,  1., -1., -0., -1., -0., -1., -0.])
>>> x0, y0, x1, y1 = g.dual.edges()
>>> x0
array([-0.707, -0.707, -0.707,  0.707,  0.707,  0.707, -1.   , -1.   ,
       -0.707, -0.707,  0.707,  0.707])
>>> x1
array([-0.707, -0.707, -0.707,  0.707,  0.707,  0.707, -0.707, -0.707,
        0.707,  0.707,  1.   ,  1.   ])
>>> y0
array([-1.   , -0.707,  0.707, -1.   , -0.707,  0.707, -0.707,  0.707,
       -0.707,  0.707, -0.707,  0.707])
>>> y1
array([-0.707,  0.707,  1.   , -0.707,  0.707,  1.   , -0.707,  0.707,
       -0.707,  0.707, -0.707,  0.707])
>>> x0, y0, x1, y1, x2, y2, x3, y3 = g.faces()
>>> x3
array([-1., -1., -0., -0.])
>>> g.dec.P(0)(lambda x, y: (x,))
array([-1., -1., -1., -0., -0., -0.,  1.,  1.,  1.])
>>> g.dec.P(0)(lambda x, y: (y,))
array([-1., -0.,  1., -1., -0.,  1., -1., -0.,  1.])
>>> g.dec.P(1)(lambda x, y: (1, 0))
array([ 1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.])
>>> g.dual.dec.P(1)(lambda x, y: (0, 1))
array([ 0.293,  1.414,  0.293,  0.293,  1.414,  0.293,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ])
>>> g.dec.P(2)(lambda x, y: (1,))
array([ 1.,  1.,  1.,  1.])
>>> g.dual.dec.P(2)(lambda x, y: (1,))
array([ 0.086,  0.414,  0.086,  0.414,  2.   ,  0.414,  0.086,  0.414,
        0.086])
>>> g = Grid_2D.chebyshev(2, 2)
>>> from spexy.spectral import to_matrix
>>> def D_matrix(d, g):
...     D = to_matrix(g.dec.D(d), g.N[d]).astype(int)
...     return D
>>> print(D_matrix(0, g))
[[-1  0  0  1  0  0  0  0  0]
 [ 0 -1  0  0  1  0  0  0  0]
 [ 0  0 -1  0  0  1  0  0  0]
 [ 0  0  0 -1  0  0  1  0  0]
 [ 0  0  0  0 -1  0  0  1  0]
 [ 0  0  0  0  0 -1  0  0  1]
 [-1  1  0  0  0  0  0  0  0]
 [ 0 -1  1  0  0  0  0  0  0]
 [ 0  0  0 -1  1  0  0  0  0]
 [ 0  0  0  0 -1  1  0  0  0]
 [ 0  0  0  0  0  0 -1  1  0]
 [ 0  0  0  0  0  0  0 -1  1]]
>>> print(D_matrix(1, g))
[[ 1 -1  0  0  0  0 -1  0  1  0  0  0]
 [ 0  1 -1  0  0  0  0 -1  0  1  0  0]
 [ 0  0  0  1 -1  0  0  0 -1  0  1  0]
 [ 0  0  0  0  1 -1  0  0  0 -1  0  1]]
>>> print(D_matrix(0, g.dual))
[[ 1  0  0  0]
 [-1  1  0  0]
 [ 0 -1  0  0]
 [ 0  0  1  0]
 [ 0  0 -1  1]
 [ 0  0  0 -1]
 [ 1  0  0  0]
 [ 0  1  0  0]
 [-1  0  1  0]
 [ 0 -1  0  1]
 [ 0  0 -1  0]
 [ 0  0  0 -1]]
>>> print(D_matrix(1, g.dual))
[[ 1  0  0  0  0  0 -1  0  0  0  0  0]
 [ 0  1  0  0  0  0  1 -1  0  0  0  0]
 [ 0  0  1  0  0  0  0  1  0  0  0  0]
 [-1  0  0  1  0  0  0  0 -1  0  0  0]
 [ 0 -1  0  0  1  0  0  0  1 -1  0  0]
 [ 0  0 -1  0  0  1  0  0  0  1  0  0]
 [ 0  0  0 -1  0  0  0  0  0  0 -1  0]
 [ 0  0  0  0 -1  0  0  0  0  0  1 -1]
 [ 0  0  0  0  0 -1  0  0  0  0  0  1]]
>>> f = g.dec.P(0)(lambda x, y: (1,))
>>> f
array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
>>> g.dec.Pup(0)(f)
(array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]),)
>>> g.refine().dec.P(0)(lambda x, y: (1,))
array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
>>> f = g.dec.P(1)(lambda x, y: (1, 0))
>>> g.dec.Pup(1)(f)
(array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]), array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]))
>>> f = g.dual.dec.P(1)(lambda x, y: (1, 0))
>>> f
array([ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.293,  0.293,
        1.414,  1.414,  0.293,  0.293])
>>> ff = g.dual.dec.Pup(1)(f)
>>> g.dual.dec.Pdown(1)(ff)
array([ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.293,  0.293,
        1.414,  1.414,  0.293,  0.293])
>>> g.refine()
Grid_2D.chebyshev(4, 4)
>>> g.refine().refine_inv()
Grid_2D.chebyshev(2, 2)
>>> f = g.proj_num(0)(lambda x, y: (x,))
>>> grad = g.grad(0)(f)
>>> grad[0]
array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
>>> grad[1] + 0.0
array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
"""
import numpy as np

from spexy.grid.dim1.samples import (cells_012, cells_345, cells_lambdify)
from spexy.grid.dim2.grid_block import GridBlock_2D
from spexy.grid.grid import Grid
from spexy.strides import (countshape, forward_idx, forward, backward)

dim = 2


class Grid_2D(Grid):
    """
    >>> g = Grid_2D.chebyshev(3, 3)
    >>> g
    Grid_2D.chebyshev(3, 3)
    """

    def __init__(self, blk):
        super().__init__()
        self.blk = blk
        assert blk.dimension == dim

    def __eq__(self, other):
        return (
            self.blk == other.blk
        )

    def __repr__(self):
        return self.blk.__repr__().replace('GridBlock_2D', 'Grid_2D')

    @property
    def dual(self):
        return type(self)(self.blk.dual)

    @classmethod
    def make(cls, name, N, M):
        return cls(GridBlock_2D.make(name, N, M))

    @property
    def dimension(self):
        return dim

    @property
    def N(self):
        return tuple(countshape(self.blk.N[d]) for d in range(dim + 1))

    def F(self, d):
        if d == 1 and self.blk._dual == 1:
            return lambda f: forward(self.blk.N[1][::-1])(f)[::-1]
        return forward(self.blk.N[d])

    def Finv(self, d):
        if d == 1 and self.blk._dual == 1:
            return lambda f: backward(self.blk.N[1][::-1])(f[::-1])
        return backward(self.blk.N[d])

    @property
    def cells(self):
        g = self.blk
        cells = unpack_cells(g.cells)
        C = [forward_idx(cells[d], g.N[d]) for d in range(dim + 1)]
        if g._dual == 1:
            C[1] = forward_idx(cells[1][::-1], g.N[1][::-1])
        return C

    @property
    def delta(self):
        g = self.blk
        delta = [forward_idx(g.delta[d], g.N[d]) for d in range(dim + 1)]
        if g._dual == 1:
            delta[1] = forward_idx(g.delta[1][::-1], g.N[1][::-1])
        return delta

    @property
    def B(self):
        g = self.blk
        B = [forward_idx(g.B[d], g.N[d]) for d in range(dim + 1)]
        if g._dual == 1:
            B[1] = forward_idx(g.B[1][::-1], g.N[1][::-1])
        return B

    def proj_num(self, d):
        return lambda f: self.Finv(d)(self.blk.proj_num(d)(f))

    def proj_sym(self, d):
        return lambda f: self.Finv(d)(self.blk.proj_sym(d)(f))

    def bndry_cond_num(self, d):
        return lambda *args: self.Finv(d)(self.blk.bndry_cond_num(d)(*args))

    def bndry_cond_sym(self, d):
        return lambda *args: self.Finv(d)(self.blk.bndry_cond_sym(d)(*args))

    def deriv(self, d):
        return lambda f: self.Finv(d + 1)(self.blk.deriv(d)(self.F(d)(f)))

    def hodge(self, d):
        return lambda f: self.dual.Finv(dim - d)(self.blk.hodge(d)(self.F(d)(f)))

    def upsample(self, d):
        return lambda f: tuple(self.refine().Finv(0)(_) for _ in self.blk.upsample(d)(self.F(d)(f)))

    def downsample(self, d):
        return lambda f: self.Finv(d)(self.blk.downsample(d)(tuple(self.refine().F(0)(_) for _ in f)))

    def grad(self, d):
        return lambda f: tuple(self.Finv(0)(_) for _ in self.blk.grad(d)(self.F(d)(f)))

    def refine(self):
        blk = self.blk.refine()
        return type(self)(blk)

    def refine_inv(self):
        blk = self.blk.refine_inv()
        return type(self)(blk)

    @classmethod
    def enum(cls, cells, numbers):
        """
        >>> c = lambda i: (i, 10*i)
        >>> n = 5
        >>> e = Grid_2D.enum(c, n)
        >>> e
        (array([0, 1, 2, 3, 4]), array([ 0, 10, 20, 30, 40]))
        """
        return np.vectorize(cells)(np.arange(numbers))


def unpack_cells(cells):
    """
    >>> import spexy.grid.dim2.grid_block as gb
    >>> cx, nx = cells_lambdify(*cells_012()[0])
    >>> cy, ny = cells_lambdify(*cells_345()[0])
    >>> cells = gb.mesh_cells(cx, cy)
    >>> C = unpack_cells(cells)
    >>> cells[0][0](0, 0)
    (0, 3)
    >>> C[0][0](0, 0)
    (0, 3)
    >>> cells[1][0](0, 0)
    (0, 1, 3)
    >>> C[1][0](0, 0)
    (0, 3, 1, 3)
    >>> cells[2][0](0, 0)
    (0, 1, 3, 4)
    >>> C[2][0](0, 0)
    (0, 3, 1, 3, 1, 4, 0, 4)
    """

    c0, c1, c2 = cells

    def vert():
        def _0(i, j):
            x, y = c0[0](i, j)
            return x, y

        return _0,

    def edge():
        def _0(i, j):
            x0, x1, y, = c1[0](i, j)
            return x0, y, x1, y

        def _1(i, j):
            x, y0, y1 = c1[1](i, j)
            return x, y0, x, y1

        return _0, _1

    def face():
        def _0(i, j):
            x0, x1, y0, y1 = c2[0](i, j)
            return x0, y0, x1, y0, x1, y1, x0, y1

        return _0,

    return vert(), edge(), face()
