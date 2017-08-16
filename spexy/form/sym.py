# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
"""
Symbolic Form

>>> f, g, u, v = sy.symbols('f, g, u, v')

>>> α = Form(0, (x, y), (f,))
>>> φ = Form(1, (x, y), (u, v))
>>> assert α ^ φ == Form(1, (x, y), (f*u, f*v))
>>> assert α ^ φ ==  φ ^ α

>>> α = Form(1, (x, y), (f, g))
>>> φ = Form(1, (x, y), (u, v))
>>> -φ
Form(1, (x, y), (-u, -v))
>>> φ + φ
Form(1, (x, y), (2*u, 2*v))
>>> φ + α
Form(1, (x, y), (f + u, g + v))
>>> φ - α
Form(1, (x, y), (-f + u, -g + v))

We can use ^ as the wedge product operator.

>>> assert φ ^ φ == Form(2, (x, y), (0,))
>>> assert α ^ φ == - φ ^ α

>>> α = Form(0, (x, y), (sy.exp(x) + sy.exp(y),))
>>> α.D.H.D.H
Form(0, (x, y), (exp(x) + exp(y),))
>>> α = Form(1, (x, y), (sy.exp(x),sy.exp(y)))
>>> α.D.H.D.H + α.H.D.H.D
Form(1, (x, y), (exp(x), exp(y)))
>>> α = Form(0, (x, y), (sy.exp(-x**2 - y**2),))
>>> sy.simplify(α.D.H.D.H.comp[0])
4*(x**2 + y**2 - 1)*exp(-x**2 - y**2)
"""
import sympy as sy

from spexy.diffgeom import PullBack, Tang
from spexy.helper import nCr
from spexy.chart import Chart

x, y, vx, vy = sy.symbols('x y vx vy')


class Form(object):
    def __init__(self, deg, xy, uv):
        self.degree = int(deg)
        self.coord = tuple(xy)
        self.comp = tuple(sy.sympify(_) for _ in uv)

        self.grid = Chart(*xy)
        self.check()

    def unwrap(self):
        deg = self.degree
        xy = self.coord
        uv = self.comp
        return deg, xy, uv

    def check(self):
        # make sure the Form has the correct number of components
        deg, xy, uv = self.unwrap()
        dim = len(xy)
        assert deg <= dim
        assert len(uv) == nCr(dim, deg)

    @classmethod
    def forms(cls, *coord):
        """
        >>> F0, F1 = Form.forms(x)
        >>> assert F0(x) == Form(0, (x,), (x,))
        >>> assert F1(x) == Form(1, (x,), (x,))
        >>> F0, F1, F2 = Form.forms(x, y)
        >>> assert F0(x * y) == Form(0, (x, y), (x * y,))
        >>> assert F1(x, y) == Form(1, (x, y), (x, y,))
        >>> assert F2(x + y) == Form(2, (x, y), (x + y,))
        """
        dim = len(coord)

        def _(deg):
            return lambda *args: cls(deg, coord, args)

        return tuple(_(deg) for deg in range(dim + 1))

    def __call__(self, *args):
        """
        >>> F0, F1, F2 = Form.forms(x, y)
        >>> F0(x - y)(x, y)
        (x - y,)
        >>> F0(x - y)(y, x)
        (-x + y,)
        >>> F0(x)(3.14, x)
        (3.14000000000000,)
        >>> sy.lambdify([x], F0(x)(3.14, x), 'numpy')(0.0)
        (3.14,)
        >>> F0(x)(1/3, x)
        (0.333333333333333,)
        >>> sy.lambdify([x], F0(x)(1/3, x), 'numpy')(0.0)
        (0.333333333333333,)
        """
        deg, xy, uv = self.unwrap()
        assert len(args) == len(xy)
        return tuple(_.subs(tuple(zip(xy, args)), simultaneous=True) for _ in uv)

    def __repr__(self):
        return 'Form{}'.format(self.unwrap().__repr__())

    def __eq__(self, other):
        return self.unwrap() == other.unwrap()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __xor__(self, other):
        return self.W(other)

    def __or__(self, other):
        return self.C(other)

    def __rmul__(self, other):
        if callable(other):
            return other(self)
        else:
            return type(self)(self.degree, (c.__rmul__(other) for c in self.comp))

    def __getitem__(self, k):
        return self.comp[k]

    @property
    def D(self):
        """
        Derivative
        """
        d, xy, uv = self.unwrap()
        uv = self.grid.D[d](uv)
        if uv is 0:
            return 0
        return Form(d + 1, xy, uv)

    @property
    def H(self):
        """
        Hodge Star
        """
        d, xy, uv = self.unwrap()
        uv = self.grid.H[d](uv)
        if uv is 0:
            return 0
        dim = len(xy)
        return Form(dim - d, xy, uv)

    def W(self, other):
        """
        Wedge Product
        """
        d1, xy1, uv1 = self.unwrap()
        d2, xy2, uv2 = other.unwrap()
        assert self.grid == other.grid
        assert xy1 == xy2
        uv = self.grid.W[d1, d2](uv1, uv2)
        return Form(d1 + d2, xy1, uv)

    def C(self, other):
        """
        Contraction
        """
        d1, xy1, uv1 = self.unwrap()
        assert d1 == 1
        d2, xy2, uv2 = other.unwrap()
        assert xy1 == xy2
        uv = self.grid.C[d2](uv1, uv2)
        if uv is 0:
            return 0
        return Form(d2 - 1, xy1, uv)

    def sharp(self):
        pass

    def flat(self):
        pass

    ###############

    def P(self, g):
        return g.P(self)

    def Ps(self, g):
        return g.Ps(self)

    ###############

    def Dbc(self, g):
        return g.BC(self)

    ################

    def lambdify(self):
        return sy.lambdify(self.coord, self.comp, 'numpy')


def binary(name):
    def _(self, other):
        if type(other) is type(self):
            assert self.degree == other.degree
            assert self.grid == other.grid
            comps = tuple(getattr(s, name)(o) for s, o in zip(self.comp, other.comp))
        else:
            comps = tuple(getattr(s, name)(other) for s in self)
        return Form(self.degree, self.coord, comps)

    return _


for name in """
        __add__
        __radd__
        __sub__
        __rsub__
        __div__
        __truediv__
        __pow__
        """.split():
    setattr(Form, name, binary(name))


def unary(name):
    def _(self):
        comps = tuple(getattr(s, name)() for s in self.comp)
        return Form(self.degree, self.coord, comps)

    return _


for name in """
        __neg__
        """.split():
    setattr(Form, name, unary(name))


def pullbackform(M):
    pull0 = PullBack(M)
    pull1 = PullBack(Tang(M))

    def pull(f):
        if f.degree == 0:
            return Form(
                0,
                [x, y],
                pull0(sy.lambdify([x, y], f.comp, 'sympy'))(x, y)
            )
        if f.degree == 1:
            p = pull1(sy.lambdify(
                [x, y, vx, vy],
                vx * f.comp[0] + vy * f.comp[1],
                'sympy'
            ))
            return Form(
                1,
                [x, y],
                [p(x, y, 1, 0), p(x, y, 0, 1)]
            )

    return pull
