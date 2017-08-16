# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from spexy.bases import regular


class BasesImp(regular.BasesImp):
    def bases(self, correct=True):
        imp = self.imp
        N, half = imp.S(self.N), imp.half

        def corr0(kappa):
            # primal boundary vertex
            if correct:
                return lambda N, n, x: kappa(N, n, x) * imp.correction0(N, n)
            return kappa

        def corrd1(kappa_star):
            # dual boundary edge
            if correct:
                return lambda N, n, x: kappa_star(N, n, x) + imp.correctiond1(N, n, x)
            return kappa_star

        kappa0 = lambda n: lambda x: corr0(imp.kappa)(N, n, x)
        kappa1 = lambda n: lambda x: imp.kappa_A_star(N, n + half, x)
        kappad0 = lambda n: lambda x: imp.kappa(N, n + half, x)
        kappad1 = lambda n: lambda x: corrd1(imp.kappa_A_star)(N, n, x)

        kappa0.grad = lambda n: lambda x: corr0(imp.kappa_grad)(N, n, x)
        kappad0.grad = lambda n: lambda x: imp.kappa_grad(N, n + half, x)

        return (kappa0, kappa1), (kappad0, kappad1)


def run_kappa():
    """
    >>> from sympy import Wild, collect, sin
    >>> from sympy.abc import x
    >>> (kappa0, kappa1), (kappad0, kappad1) = BasesImp(2, 'sym').bases()
    >>> kappa1(0)(x)
    sin(x)/2 + sin(2*x)/2
    >>> kappa1(1)(x)
    sin(x)/2 - sin(2*x)/2

    >>> X = Wild('X')
    >>> c = lambda expr: collect(expr.simplify(), sin(X))
    >>> kappad1(0)(x)
    sin(x)/4 + sin(2*x) + 3*sin(3*x)/4
    >>> c( kappad1(1)(x) )
    (1/4 + sqrt(2)/4)*sin(x) + (-3*sqrt(2)/4 + 3/4)*sin(3*x)
    >>> kappad1(2)(x)
    sin(x)/4 - sin(2*x) + 3*sin(3*x)/4
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
