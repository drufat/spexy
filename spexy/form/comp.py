# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
"""
Component Form

>>> from spexy.grid import Grid_1D, Grid_2D, GridBlock_2D

>>> g = Grid_1D.chebyshev(2)
>>> a = g.P(0, lambda x: 1)
>>> b = g.P(1, lambda x: 1)
>>> a = a.Pup
>>> b = b.Pup
>>> a + 1
Form(0, (Form(0, Grid_1D.chebyshev(4), array([ 2.,  2.,  2.,  2.,  2.])),))
>>> w = a ^ b
>>> w.Pdown(g).array
array([ 1.,  1.])
>>> i = b.C(b)
>>> i.Pdown(g).array
array([ 1.,  1.,  1.])
>>> a.Pup.shape()
((9,),)
>>> a.Pup.Pup.shape()
((17,),)
>>> a.Pup.Pup.Pdown_().shape()
((9,),)
>>> g = Grid_2D.chebyshev(2, 2)
>>> a = g.P(1, lambda x, y: (1, 0))
>>> a
Form(1, Grid_2D.chebyshev(2, 2), array([ 1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]))
>>> a.Pup.comp[0].array
array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
>>> a.Pup.comp[1].array
array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
>>> a.Pup.Pdown(g) == a
True
>>> type(a)
<class 'spexy.form.coch.Form'>
>>> a = a.Pup
>>> a.comp[0].grid.dimension
2
>>> a.shape()
((25,), (25,))
>>> a.Pup.shape()
((81,), (81,))
>>> a.Pup.Pup.shape()
((289,), (289,))
>>> a.Pup.Pup.Pdown_().shape()
((81,), (81,))
>>> s = a + a
>>> s.Pdown(g).array
array([ 2.,  2.,  2.,  2.,  2.,  2.,  0.,  0.,  0.,  0.,  0.,  0.])
>>> s = a + (0, -1)
>>> s.Pdown(g).array
array([ 1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1., -1.])
>>> w = a ^ a
>>> w.Pdown(g).array
array([ 0.,  0.,  0.,  0.])
>>> i = a.C(a)
>>> i.Pdown(g).array
array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])

>>> g = GridBlock_2D.chebyshev(2, 2)
>>> a = g.P(1, lambda x, y: (1, 0))
>>> a.Pup.shape()
((1, 5, 5), (1, 5, 5))
>>> a
Form(1, GridBlock_2D.chebyshev(2, 2), array([array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]]),
       array([[ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.]])], dtype=object))
>>> a.Pup.Pdown(g)
Form(1, GridBlock_2D.chebyshev(2, 2), array([array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]]),
       array([[ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.]])], dtype=object))
>>> a.Pup.B([0, 1, -1, 0], [0, -1, 1, 0]).Pdown(g)
Form(1, GridBlock_2D.chebyshev(2, 2), array([array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.]]),
       array([[-1., -1.],
       [-1., -1.],
       [-1., -1.]])], dtype=object))
"""

from collections import Iterable

import spexy.chart
from spexy.form import coch
from spexy.helper import nCr


class Form(object):
    def __init__(self, degree, comp):
        self.degree = degree
        self.comp = tuple(comp)  # tuple of dec zero forms
        self.check()

    def unwrap(self):
        deg = self.degree
        comp = self.comp
        return deg, comp

    def check(self):
        # make sure the Form has the correct number of components
        deg, comp = self.unwrap()
        g = comp[0].grid
        dim = g.dimension
        for c in comp:
            assert c.grid == g
            assert c.degree == 0
        assert abs(deg) <= dim
        assert len(comp) == nCr(dim, abs(deg))

    def shape(self):
        return tuple(c.array.shape for c in self.comp)

    def array(self):
        return tuple(_.array for _ in self.comp)

    def __repr__(self):
        d, c = self.unwrap()
        return 'Form{}'.format((d, c).__repr__())

    def __eq__(self, other):
        return (
            self.degree == other.degree
            and
            self.comp == other.comp
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __xor__(self, other):
        return self.W(other)

    def __or__(self, other):
        return self.C(other)

    @property
    def H(self):
        """
        >>> from spexy.grid import Grid_1D, Grid_2D
        >>> g = Grid_1D.chebyshev(5)
        >>> a = g.P(0, lambda x: 1)
        >>> assert a.Pup.H.Pdown(g.dual) == a.H
        >>> g = Grid_2D.chebyshev(5, 5)
        >>> a = g.P(1, lambda x, y: (-y, x))
        >>> assert a.Pup.H.Pdown(g.dual) == a.H
        >>> a = g.dual.P(0, lambda x, y: (x - y,))
        >>> assert a.Pup.H.Pdown(g) == a.H
        """
        d, c = self.unwrap()
        g = c[0].grid

        if g.dimension == 1:
            H = spexy.chart.hodge_star_1d()
        elif g.dimension == 2:
            H = spexy.chart.hodge_star_2d()
        else:
            raise NotImplementedError
        dim = g.dimension
        c = H[d](c)
        return Form(dim - d, c)

    def W(self, other):
        d1, c1 = self.unwrap()
        d2, c2 = other.unwrap()
        g1 = c1[0].grid
        g2 = c2[0].grid
        assert g1 == g2
        assert d1 + d2 <= g1.dimension

        if g1.dimension == 1:
            W = spexy.chart.wedge_1d()
        elif g1.dimension == 2:
            W = spexy.chart.wedge_2d()
        else:
            raise NotImplementedError

        c = W[d1, d2](c1, c2)
        return Form(d1 + d2, c)

    def B(self, J, Jinv):
        d, c = self.unwrap()
        g = c[0].grid
        J11, J12, J21, J22 = J
        Ji11, Ji12, Ji21, Ji22 = Jinv
        if d == 0:
            return self
        elif d == 1:
            a1, a2 = [_.array for _ in c]
            a1, a2 = [
                a1 * Ji11 + a2 * Ji21,
                a1 * Ji12 + a2 * Ji22,
            ]
            return Form(d, [
                coch.Form(0, g, a1),
                coch.Form(0, g, a2),
            ])
        elif d == -1:
            a1, a2 = [_.array for _ in c]
            a1, a2 = [
                J11 * a1 + J12 * a2,
                J21 * a1 + J22 * a2,
            ]
            return Form(d, [
                coch.Form(0, g, a1),
                coch.Form(0, g, a2),
            ])
        elif d == 2:
            a, = [_.array for _ in c]
            a = a * (Ji11 * Ji22 - Ji21 * Ji12)
            return Form(d, [
                coch.Form(0, g, a),
            ])
        else:
            raise NotImplementedError

    ##################
    # J must lead to an orthonormal frame

    def flat(self, J, Jinv):
        assert self.degree == -1
        return Form(+1, self.B(J, Jinv).comp).B(Jinv, J)

    def sharp(self, J, Jinv):
        assert self.degree == +1
        return Form(-1, self.B(J, Jinv).comp).B(Jinv, J)

    ###############

    def C(self, other):
        d1, c1 = self.unwrap()
        d2, c2 = other.unwrap()
        g1 = c1[0].grid
        g2 = c2[0].grid
        assert g1 == g2
        assert d1 == 1

        if g1.dimension == 1:
            C = spexy.chart.contraction_1d()
        elif g1.dimension == 2:
            C = spexy.chart.contraction_2d()
        else:
            raise NotImplementedError

        c = C[d2](c1, c2)
        return Form(d2 - 1, c)

    ###########
    def Pdown(self, to):
        return to.Pdown(self)

    ###########
    @property
    def Pup(self):
        d, c = self.unwrap()
        cc = tuple(_.Pup.comp[0] for _ in c)
        return Form(d, cc)

    def Pdown_(self):
        """
        >>> from spexy.grid import Grid_2D
        >>> g = Grid_2D.chebyshev(2, 2)
        >>> a = g.P(1, lambda x, y: (1, 0))
        >>> a.Pup == a.Pup.Pup.Pdown_()
        True

        >>> from spexy.grid import GridBlock_2D
        >>> g = GridBlock_2D.chebyshev(2, 2)
        >>> a = g.P(1, lambda x, y: (1, 0))
        >>> a.Pup.Pup.Pdown_() == a.Pup
        True
        """
        d, cc = self.unwrap()
        g = cc[0].grid.refine_inv()
        c = [Form(0, [_]).Pdown(g) for _ in cc]
        return Form(d, c)


def binary(name):
    def _(self, other):
        if type(other) is type(self):
            assert self.degree == other.degree
            assert self.shape() == other.shape()
            comp = tuple(getattr(s, name)(o) for s, o in zip(self.comp, other.comp))
        elif isinstance(other, Iterable):
            comp = tuple(getattr(s, name)(o) for s, o in zip(self.comp, other))
        else:
            comp = tuple(getattr(s, name)(other) for s in self.comp)
        return Form(self.degree, comp)

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
        comp = tuple(getattr(s, name)() for s in self.comp)
        return Form(self.degree, comp)

    return _


for name in """
        __neg__
        """.split():
    setattr(Form, name, unary(name))
