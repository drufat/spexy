# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
"""
DEC form that lives on cells

>>> from spexy.grid import Grid_1D, Grid_2D, GridBlock_2D

>>> g = Grid_1D.chebyshev(2)
>>> α = Form(0, g, [1, 1, 1])
>>> β = Form(1, g, [1, 1])
>>> α ^ β
Form(1, Grid_1D.chebyshev(2), array([ 1.,  1.]))
>>> α.D
Form(1, Grid_1D.chebyshev(2), array([0, 0]))
>>> α.G + 0.0
Form(1, (Form(0, Grid_1D.chebyshev(2), array([ 0.,  0.,  0.])),))
>>> from sympy.abc import x, y
>>> from spexy.form.sym import Form
>>> F0, F1 = Form.forms(x)
>>> F0(1)
Form(0, (x,), (1,))
>>> g.P(F0(1))
Form(0, Grid_1D.chebyshev(2), array([ 1.,  1.,  1.]))
>>> g.P(F1(1))
Form(1, Grid_1D.chebyshev(2), array([ 1.,  1.]))

>>> g = Grid_2D.chebyshev(2, 2)
>>> F0, F1, F2 = Form.forms(x, y)
>>> g.P(F0(1))
Form(0, Grid_2D.chebyshev(2, 2), array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]))

>>> g = GridBlock_2D.chebyshev(2, 2)
>>> g.P(F0(1))
Form(0, GridBlock_2D.chebyshev(2, 2), array([[[ 1.,  1.,  1.],
        [ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]]))
>>> g.P(F1(1, 0))
Form(1, GridBlock_2D.chebyshev(2, 2), array([array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]]),
       array([[ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.]])], dtype=object))
>>> g.P(F2(1))
Form(2, GridBlock_2D.chebyshev(2, 2), array([[[ 1.,  1.],
        [ 1.,  1.]]]))
"""
import numpy as np


class Form(object):
    def __init__(self, degree, grid, array):
        self.degree = degree
        self.grid = grid
        self.array = np.asarray(array)

        self.check()

    def unwrap(self):
        d, g, a = self.degree, self.grid, self.array
        return d, g, a

    def check(self):
        d, g, a = self.unwrap()
        assert d <= g.dimension
        if type(g.N[d]) is int:
            assert a.shape == (g.N[d],)

    def __repr__(self):
        d, g, a = self.unwrap()
        return "Form{}".format((d, g, a).__repr__())

    def __eq__(self, other):
        return (
            self.degree == other.degree and
            self.grid == other.grid and
            np.allclose(self.array, other.array)
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __xor__(self, other):
        return self.W(other)

    def __or__(self, other):
        return self.C(other)

    @property
    def D(self):
        return self.grid.D(self)

    @property
    def H(self):
        return self.grid.H(self)

    @property
    def G(self):
        return self.grid.G(self)

    def W(self, other, to=None):
        """
        Wedge Product
        """
        d1, g1, a1 = self.unwrap()
        d2, g2, a2 = other.unwrap()
        g3 = (to if to else g2)
        assert g1 == g2 or g1 == g2.dual
        if d1 == d2 == 0 and g1 == g2 == g3:
            # no refinement necessary, just multiply directly
            return Form(d1 + d2, g2, a1 * a2)
        return (self.Pup ^ other.Pup).Pdown(g3)

    def C(self, other, to=None):
        """
        Contraction
        """
        d1, g1, a1 = self.unwrap()
        d2, g2, a2 = other.unwrap()
        g3 = (to if to else g2)
        assert d1 == 1
        assert g1 == g2 or g1 == g2.dual
        return (self.Pup | other.Pup).Pdown(g3)

    #############

    @property
    def Lap(self):
        return self.D.H.D.H + self.H.D.H.D

    def Lie(self, other):
        return (self.C(other)).D + self.C(other.D)

    #############

    @property
    def Pup(self):
        """
        Cochain Form -> Component Form
        """
        return self.grid.Pup(self)

    #############
    @property
    def R(self):
        """
        Reconstruction
        Discrete Form -> Python Function
        """
        d, g, a = self.unwrap()
        B = g.B

        def func(*x):
            return sum(a[i] * B[d](i)(*x) for i in range(g.N[d]))

        return func

    @property
    def Rf(self):
        d, g, a = self.unwrap()
        return g.dec.Pup[d](a)

    def HH(self, J, Jinv):
        return self.Pup.B(J, Jinv).H.B(Jinv, J).Pdown(self.grid.dual)

    @property
    def blk(self):
        d, g, a = self.unwrap()
        return Form(d, g.blk, g.F(d)(a))

    @property
    def blkinv(self):
        d, g, a = self.unwrap()
        from spexy.grid.dim2.grid import Grid_2D
        g = Grid_2D(g)
        return Form(d, g, g.Finv(d)(a))


def binary(name):
    def _(self, other):
        if type(other) == type(self):
            assert self.degree == other.degree
            assert self.grid == other.grid or self.grid == other.grid.dual
            array = getattr(self.array, name)(other.array)
        else:
            array = getattr(self.array, name)(other)
        return Form(self.degree, self.grid, array)

    return _


for name in """
        __add__
        __radd__
        __mul__
        __rmul__
        __sub__
        __rsub__
        __div__
        __truediv__
        __pow__
        """.split():
    setattr(Form, name, binary(name))


def unary(name):
    def _(self):
        array = getattr(self.array, name)()
        return Form(self.degree, self.grid, array)

    return _


for name in """
        __neg__
        """.split():
    setattr(Form, name, unary(name))
