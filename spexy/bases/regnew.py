# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from spexy.bases import regular


class BasesImp(regular.BasesImp):
    def numbers(self):
        N = self.N

        N0 = N - 1
        N1 = N
        N0d = N
        N1d = N - 1
        return (N0, N1), (N0d, N1d)

    def cells_index(self):
        half = self.imp.half

        i0 = lambda n: (n + 1,)
        i1 = lambda n: (n, n + 1)
        id0 = lambda n: (n + half,)
        id1 = lambda n: (n + half, n + 3 * half)
        return (i0, i1), (id0, id1)

    def points(self, n):
        N = self.N
        return self.imp.points_regular(N, n)

    def bases(self, correct=True):
        imp = self.imp
        N, half = imp.S(self.N), imp.half
        pi = imp.pi
        f = N / (N - 1)

        def s(x): return (x - pi / 2) * f + pi / 2

        # Bases Functions
        kappa0 = lambda n: lambda x: imp.kappa(N - 1, n + half, s(x))
        kappa1 = lambda n: lambda x: imp.kappa_star(N, n + half, x)
        kappad0 = lambda n: lambda x: imp.kappa(N, n + half, x)
        kappad1 = lambda n: lambda x: imp.kappa_star(N - 1, n + half, s(x)) * f

        # Gradients
        kappa0.grad = lambda n: lambda x: imp.kappa_grad(N - 1, n + half, s(x))
        kappad0.grad = lambda n: lambda x: imp.kappa_grad(N, n + half, x)

        return (kappa0, kappa1), (kappad0, kappad1)

    def boundary(self):
        pi = self.imp.pi
        return (0, pi), None


def run(N):
    """
    >>> run(2)
    zero-form
    [1]
    one-form
    [1, 0]
    [0, 1]
    dual zero-form
    [1, 0]
    [0, 1]
    dual one-form
    [1]
    >>> run(3)
    zero-form
    [1, 0]
    [0, 1]
    one-form
    [1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]
    dual zero-form
    [1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]
    dual one-form
    [1, 0]
    [0, 1]
    """
    from spexy.bases.symintegrals import run_integrals
    run_integrals(BasesImp)(N)
