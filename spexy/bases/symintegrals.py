# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from spexy.symbolic import replace_piecewise_default
from sympy import (exp, cos, I, sin, Wild, collect, symbols, cot, tan, sqrt, fraction, factor, div, poly, integrate,
                   simplify)
from sympy.abc import x


def mirror(phi, weight=(lambda _: 1)):
    return lambda n, x: (
        + (phi(n, +x) / weight(+x))
        + (phi(n, -x) / weight(-x))
    )


def kappa_simplify(expr):
    expr = replace_piecewise_default(expr)

    expr = expr.doit().expand(trig=True)
    expr = expr.rewrite(exp)
    expr = expr.expand().cancel().expand()
    expr = expr.replace(exp, lambda x: cos(x / I) + I * sin(x / I))
    expr = expr.cancel()
    X = Wild('X')
    expr = collect(expr, [sin(X), cos(X)])
    return expr


def psi_simplify(expr):
    X = Wild('X')
    x = symbols('x')
    u = symbols('u', positive=True)

    expr = replace_piecewise_default(expr)

    expr = expr.doit().expand(trig=True)
    expr = expr.replace(cot(X), 1 / tan(X)).replace(tan(X), sin(X) / cos(X))
    expr = expr.replace(cos(X / 2), sqrt((1 + cos(X)) / 2)).replace(sin(X / 2), sqrt((1 - cos(X)) / 2))
    expr = expr.subs(x, 1 - u).simplify().subs(u, 1 - x).expand()

    num, denom = fraction(factor(expr))
    if denom == 1:
        expr = num.expand()
    else:
        q, r = div(poly(num, x), poly(denom, x))
        assert r == 0
        expr = q.as_expr()

    return collect(expr, x)


def run_integrals(BasesImp):
    def run(N):
        b = BasesImp(N, imp='sym')
        (c0, c1), (cd0, cd1) = b.cells()
        (b0, b1), (bd0, bd1) = b.bases()
        (n0, n1), (nd0, nd1) = b.numbers()

        print('zero-form')
        for j in range(n0):
            expr = b0(j)(x)
            rslt = [simplify(expr.subs(x, c0(i)[0]))
                    for i in range(n0)]
            print(rslt)

        print('one-form')
        for j in range(n1):
            expr = b1(j)(x)
            expr = integrate(expr, x)
            rslt = [simplify(expr.subs(x, c1(i)[1]) -
                             expr.subs(x, c1(i)[0]))
                    for i in range(n1)]
            print(rslt)

        print('dual zero-form')
        for j in range(nd0):
            expr = bd0(j)(x)
            rslt = [simplify(expr.subs(x, cd0(i)[0]))
                    for i in range(nd0)]
            print(rslt)

        print('dual one-form')
        for j in range(nd1):
            expr = bd1(j)(x)
            expr = integrate(expr, x)
            rslt = [simplify(expr.subs(x, cd1(i)[1]) -
                             expr.subs(x, cd1(i)[0]))
                    for i in range(nd1)]
            print(rslt)

    return run
