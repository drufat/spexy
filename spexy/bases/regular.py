# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from spexy.bases import basesimp


class BasesImp(basesimp.BasesImp):
    def module(self):
        return 'spexy.bases.circular'

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
        N = self.N
        return self.imp.points_regular_clamped(N, n)

    def bases(self, correct=True):
        imp = self.imp
        N, half = imp.S(self.N), imp.half

        def corr0(kappa):
            # primal boundary vertex
            if correct:
                return lambda N, n, x: kappa(N, n, x) * imp.correction0(N, n)
            return kappa

        # Bases Functions
        kappa0 = lambda n: lambda x: corr0(imp.kappa)(N, n, x)
        kappa1 = lambda n: lambda x: imp.kappa_star(N, n + half, x)
        kappad0 = lambda n: lambda x: imp.kappa(N, n + half, x)
        kappad1 = lambda n: lambda x: imp.kappa_star(N, n, x)

        # Gradients
        kappa0.grad = lambda n: lambda x: corr0(imp.kappa_grad)(N, n, x)
        kappad0.grad = lambda n: lambda x: imp.kappa_grad(N, n + half, x)

        return (kappa0, kappa1), (kappad0, kappad1)

    def boundary(self):
        pi = self.imp.pi
        return None, (0, pi)


def run_kappa():
    """
    >>> from sympy.abc import x
    >>> (kappa0, kappa1), (kappad0, kappad1) = BasesImp(2, 'sym').bases()
    >>> kappa0(0)(x)
    cos(x)/2 + cos(2*x)/4 + 1/4
    >>> kappa0(1)(x)
    -cos(2*x)/2 + 1/2
    >>> kappa0(2)(x)
    -cos(x)/2 + cos(2*x)/4 + 1/4

    >>> kappa1(0)(x)
    cos(x)/2 + 1/pi
    >>> kappa1(1)(x)
    -cos(x)/2 + 1/pi

    >>> kappad0(0)(x)
    sqrt(2)*cos(x)/2 + 1/2
    >>> kappad0(1)(x)
    -sqrt(2)*cos(x)/2 + 1/2

    >>> kappad1(0)(x)
    sqrt(2)*cos(x)/2 + cos(2*x)/2 + 1/pi
    >>> kappad1(1)(x)
    -cos(2*x)/2 + 1/pi
    >>> kappad1(2)(x)
    -sqrt(2)*cos(x)/2 + cos(2*x)/2 + 1/pi
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

    >>> run(3)
    zero-form
    [1, 0, 0, 0]
    [0, 1, 0, 0]
    [0, 0, 1, 0]
    [0, 0, 0, 1]
    one-form
    [1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]
    dual zero-form
    [1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]
    dual one-form
    [1, 0, 0, 0]
    [0, 1, 0, 0]
    [0, 0, 1, 0]
    [0, 0, 0, 1]
    """
    from spexy.bases.symintegrals import run_integrals
    run_integrals(BasesImp)(N)
