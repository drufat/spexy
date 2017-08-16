# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import importlib
import itertools
import types

import numpy as np
from spexy.form import (coch, comp, sym)
from spexy.helper import bunch
from spexy.grid.gridhelper import GridHelper


class Grid(GridHelper):
    def __init__(self):
        self.dec = bunch(
            P=self.proj_num,
            Ps=self.proj_sym,
            BC=self.bndry_cond_num,
            BCs=self.bndry_cond_sym,
            D=self.deriv,
            H=self.hodge,
            Pup=self.upsample,
            Pdown=self.downsample,
            G=self.grad,
        )

    def rand(self, deg):
        """
        Create a random Form.
        """
        return coch.Form(
            deg,
            self,
            np.random.rand(self.N[deg])
        )

    def zero(self, deg):
        """
        >>> from spexy.grid import Grid_1D, Grid_2D
        >>> g = Grid_1D.chebyshev(2)
        >>> g.zero(0)
        Form(0, Grid_1D.chebyshev(2), array([ 0.,  0.,  0.]))
        >>> g = Grid_2D.chebyshev(2, 2)
        >>> g.zero(1)
        Form(1, Grid_2D.chebyshev(2, 2), array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]))
        """
        return coch.Form(
            deg,
            self,
            np.zeros(self.N[deg], dtype=np.float)
        )

    def one(self, deg):
        return coch.Form(
            deg,
            self,
            np.ones([self.N[deg]], dtype=np.float)
        )

    def Ps(self, form):
        """
        >>> from spexy.continuous import F0, F1, x, y
        >>> f = F1(-y, x)
        >>> coord = f.grid.coord
        >>> comps = F1(-y, x).comp
        >>> xy = (x, y)
        >>> tuple(_.subs([(x, 2), (y, 0)]) for _ in comps)
        (0, 2)
        >>> fl = lambda *xy: tuple(_.subs(tuple(zip(coord, xy))) for _ in comps)
        >>> fl(2, 1)
        (-1, 2)
        """
        func = form
        if form.grid.dim == 1:
            func = lambda *xy: form(*xy)[0]
        return coch.Form(
            form.degree,
            self,
            self.proj_sym(form.degree)(func)
        )

    def P(self, *args):
        """
        >>> from spexy.grid import Grid_1D
        >>> from spexy.form.sym import Form, x
        >>> F0, F1 = Form.forms(x)
        >>> g = Grid_1D.chebyshev(2)
        >>> f1 = g.P(0, lambda x: x)
        >>> f1
        Form(0, Grid_1D.chebyshev(2), array([-1., -0.,  1.]))
        >>> f2 = g.P(F0(x))
        >>> f3 = g.Ps(F0(x))
        >>> f1 == f2 == f3
        True
        """
        args_t = tuple(type(_) for _ in args)

        if args_t == (int, types.FunctionType):
            degree, func = args
            return coch.Form(
                degree,
                self,
                self.proj_num(degree)(func)
            )

        if args_t == (sym.Form,):
            form, = args
            func = form.lambdify()
            if form.grid.dim == 1:
                func_ = func
                func = lambda *xy: func_(*xy)[0]
            return self.P(form.degree, func)

        raise TypeError(args)

    def BCs(self, form):
        func = form
        if form.grid.dim == 1:
            func = lambda *xy: form(*xy)[0]
        degree = form.degree + 1
        return coch.Form(
            degree,
            self,
            self.bndry_cond_sym(degree)(func)
        )

    def BC(self, *args):
        """
        >>> from spexy.grid import Grid_1D
        >>> from spexy.form.sym import Form, x
        >>> F0, F1 = Form.forms(x)
        >>> g = Grid_1D.chebyshev(2)
        >>> f1 = g.dual.BC(0, lambda x: x)
        >>> f1
        Form(1, Grid_1D.chebyshev(2).dual, array([ 1.,  0.,  1.]))
        >>> f2 = g.dual.BC(F0(x))
        >>> f3 = g.dual.BCs(F0(x))
        >>> f1 == f2 == f3
        True
        """
        args_t = tuple(type(_) for _ in args)

        if args_t == (int, types.FunctionType):
            degree, func = args
            degree += 1
            return coch.Form(
                degree,
                self,
                self.bndry_cond_num(degree)(func)
            )

        if args_t == (int, types.FunctionType, types.FunctionType, types.FunctionType, types.FunctionType):
            degree, fxmin, fxmax, fymin, fymax = args
            degree += 1
            return coch.Form(
                degree,
                self,
                self.bndry_cond_num(degree)(fxmin, fxmax, fymin, fymax)
            )

        if args_t == (sym.Form,):
            form, = args
            func = form.lambdify()
            if form.grid.dim == 1:
                func_ = func
                func = lambda *xy: func_(*xy)[0]
            return self.BC(form.degree, func)

        raise TypeError(args)

    def D(self, f):
        """
        Derivative
        """
        d, g, a = f.unwrap()
        assert g == self
        return type(f)(
            d + 1,
            g,
            g.deriv(d)(a)
        )

    def H(self, f):
        """
        Hodge Star
        """
        d, g, a = f.unwrap()
        assert g == self
        return type(f)(
            g.dimension - d,
            g.dual,
            g.hodge(d)(a)
        )

    def Pup(self, f):
        """
        DEC Form -> Component Form
        """
        d, g, a = f.unwrap()
        assert g == self or g == self.dual
        a = g.upsample(d)(a)
        if g.dimension == 1:
            a = [a]
        gg = self.refine()
        c = tuple(type(f)(0, gg, _) for _ in a)
        return comp.Form(d, c)

    def Pdown(self, f):
        """
        Component Form -> DEC Form
        """
        d, c = f.unwrap()
        a = tuple(_.array for _ in c)
        if self.dimension == 1:
            [a] = a
        a = self.downsample(d)(a)
        return type(c[0])(d, self, a)

    def G(self, f):
        """
        Gradient
        """
        d, g, a = f.unwrap()
        assert g == self
        assert d == 0
        a = g.grad(d)(a)
        if g.dimension == 1:
            a = [a]
        c = tuple(type(f)(0, g, _) for _ in a)
        return comp.Form(1, c)


names = [
    'periodic',
    'regular',
    'antiregular',
    'regnew',
    'chebyshev',
    'chebold',
    'chebnew',
]


def setconstructor(name):
    @classmethod
    def _(cls, *args):
        return cls.make(name, *args)

    setattr(Grid, name, _)


[setconstructor(name) for name in names]
bases = {name: importlib.import_module('spexy.bases.{}'.format(name)) for name in names}
grids = {name: importlib.import_module('spexy.types.{}'.format(name)) for name in names}


def wrap(*args):
    """
    >>> P0, P1, P2, P0d, P1d, P2d = range(6)
    >>> wrap(P0, P1, P0d, P1d) == {
    ...    (0, True): P0,
    ...    (1, True): P1,
    ...    (0, False): P0d,
    ...    (1, False): P1d,
    ... }
    True
    >>> wrap(P0, P1, P2, P0d, P1d, P2d) == {
    ...    (0, True): P0,
    ...    (1, True): P1,
    ...    (2, True): P2,
    ...    (0, False): P0d,
    ...    (1, False): P1d,
    ...    (2, False): P2d,
    ... }
    True
    """
    N = len(args)
    assert (N % 2 == 0)
    primal = args[:N // 2]
    dual = args[N // 2:]
    rslt = {}
    rslt.update({(i, True): _ for i, _ in enumerate(primal)})
    rslt.update({(i, False): _ for i, _ in enumerate(dual)})
    return rslt


def unwrap(arg):
    """
    >>> P0, P1, P2, P0d, P1d, P2d = range(6)
    >>> unwrap({
    ...    (0, True): P0,
    ...    (1, True): P1,
    ...    (0, False): P0d,
    ...    (1, False): P1d,
    ... }) == (P0, P1, P0d, P1d)
    True
    >>> unwrap({
    ...    (0, True): P0,
    ...    (1, True): P1,
    ...    (2, True): P2,
    ...    (0, False): P0d,
    ...    (1, False): P1d,
    ...    (2, False): P2d,
    ... }) == (P0, P1, P2, P0d, P1d, P2d)
    True
    """
    assert type(arg) is dict
    N = len(arg)
    assert (N % 2 == 0)
    primal = tuple(arg[i, True] for i in range(N // 2))
    dual = tuple(arg[i, False] for i in range(N // 2))
    return primal + dual


def Π(*x):
    """
    >>> Π((0, 1, 2), 'ab')
    ((0, 'a'), (0, 'b'), (1, 'a'), (1, 'b'), (2, 'a'), (2, 'b'))
    """
    return tuple(itertools.product(*x))


def reconstruction(B, N):
    """
    Give the reconstruction functions for the set of basis functions basis_fn.
    """

    def _(f):
        return lambda *x: sum(f[i] * B(i)(*x) for i in range(N))

    return _


def derivative_matrix(g):
    """
    Compute Derivative matrix from basis functions.
    Special care must be taken for boundary conditions.
    """

    def _(g):
        B = g.B[0].grad
        N = g.N[0]
        P = g.dec.P(1)
        H = np.vstack(P(B(i)) for i in range(N)).T
        return H

    return _(g), _(g.dual)


def hodge_star_matrix(g):
    """
    Compute Hodge-Star matrix from basis functions.
    """
    n = g.dimension

    def _(g, d):
        B = g.B[d]
        N = g.N[d]
        P = g.dual.dec.P(n - d)
        H = np.vstack(P(B(i)) for i in range(N)).T
        return H

    return _(g, 0), _(g, 1), _(g.dual, 0), _(g.dual, 1)


def switch_matrix(g):
    assert g.dimension == 1

    B = g.B[0]
    N = g.N[0]
    P = g.dual.dec.P(0)
    S = np.vstack(P(B(i)) for i in range(N)).T

    B = g.dual.B[0]
    N = g.dual.N[0]
    P = g.dec.P(0)
    Sinv = np.vstack(P(B(i)) for i in range(N)).T

    return S, Sinv


def qswitch_matrix(g):
    assert g.dimension == 1

    B = g.B[0]
    N = g.N[0]
    P = g.dec.P(1)
    Q = np.vstack(P(B(i)) for i in range(N)).T

    B = g.B[1]
    N = g.N[1]
    P = g.dec.P(0)
    Qinv = np.vstack(P(B(i)) for i in range(N)).T

    return Q, Qinv


def upsample_matrix(g, gg):
    assert g.dimension == gg.dimension == 1
    P = gg.dec.P(0)

    def _(g, d):
        B = g.B[d]
        N = g.N[d]
        T = np.vstack(P(B(i)) for i in range(N)).T
        return T

    return _(g, 0), _(g, 1), _(g.dual, 0), _(g.dual, 1)


def downsample_matrix(gg, g):
    assert g.dimension == gg.dimension == 1
    B = gg.B[0]
    N = gg.N[0]

    def _(g, d):
        P = g.dec.P(d)
        U = np.vstack(P(B(i)) for i in range(N)).T
        return U

    return _(g, 0), _(g, 1), _(g.dual, 0), _(g.dual, 1)


def gradient_matrix(g):
    def _(g):
        B = g.B[0].grad
        N = g.N[0]
        P = g.dec.P(0)
        G = np.vstack(P(B(i)) for i in range(N)).T
        return G

    return _(g), _(g.dual)
