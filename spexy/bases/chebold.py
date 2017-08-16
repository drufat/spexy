# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from spexy.bases import regnew


class BasesImp(regnew.BasesImp):
    def module(self):
        return 'spexy.bases.cardinals'

    def numbers(self):
        N = self.N

        N0 = N + 1
        N1 = N
        N0d = N
        N1d = N + 1
        return (N0, N1), (N0d, N1d)

    def cells_index(self):
        half = self.imp.half

        i0 = lambda n: (n,)
        i1 = lambda n: (n, n + 1)
        id0 = lambda n: (n + half,)
        id1 = lambda n: (n - half, n + half)
        return (i0, i1), (id0, id1)

    def points(self, n):
        imp = self.imp
        N = imp.S(self.N)

        return imp.xUclamp(N + 1, n)

    def bases(self):
        imp = self.imp
        N = imp.S(self.N)

        psi0 = lambda n: lambda x: imp.CUclamp(N + 1, n, x)
        psi1 = lambda n: lambda x: imp.DUclamp(N + 1, n, x)
        psid0 = lambda n: lambda x: imp.CT(N, n, x)
        psid1 = lambda n: lambda x: imp.DTclamp(N + 2, n, x)

        psi0.grad = lambda n: lambda x: imp.dCUclamp(N + 1, n, x)
        psid0.grad = lambda n: lambda x: imp.dCT(N, n, x)

        return (psi0, psi1), (psid0, psid1)

    def boundary(self):
        return None, (-1, +1)


def run_psi():
    """
    >>> import sympy as sy
    >>> from sympy.abc import x
    >>> half = sy.Rational(1, 2)

    >>> b = BasesImp(2, 'sym')
    >>> (psi0, psi1), (psid0, psid1) = b.bases()
    >>> p = b.points
    >>> p(0), p(1), p(2)
    (-1, 0, 1)
    >>> p(-half), p(half), p(1 + half), p(2+half)
    (-1, -sqrt(2)/2, sqrt(2)/2, 1)
    >>> psi0(0)(x).subs(x, p(0))
    1
    >>> psid0(0)(x).subs(x, p(half))
    1
    >>> sy.integrate(psid1(0)(x), (x, p(-half), p(half)))
    1
    >>> sy.integrate(psid1(0)(x), (x, p(half), p(1+half)))
    0
    >>> sy.integrate(psi1(0)(x), (x, p(0), p(1)))
    1
    >>> sy.integrate(psi1(0)(x), (x, p(1), p(2)))
    0
    """
    pass


def run(N):
    """
    >>> run(1)
    zero-form
    [1, 0]
    [0, 1]
    one-form
    [1]
    dual zero-form
    [1]
    dual one-form
    [1, 0]
    [0, 1]
    >>> run(2)
    zero-form
    [1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]
    one-form
    [1, 0]
    [0, 1]
    dual zero-form
    [1, 0]
    [0, 1]
    dual one-form
    [1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]
    """
    from spexy.bases.symintegrals import run_integrals
    run_integrals(BasesImp)(N)
