# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np
from spexy.other.sympy import lambdify, Printer

from sympy import symbols, Sum, Equality, Unequality
from sympy.abc import x


# test_numpy.py

def test_sum():
    k, k0, kN = symbols("k, k0, kN")

    s = Sum(x ** k, (k, k0, kN))
    f = lambdify((k0, kN, x), s, 'numpy')

    k0_, kN_ = 0, 10
    x_ = np.linspace(-1, +1, 100)
    assert np.allclose(f(k0_, kN_, x_), sum(x_ ** k_ for k_ in range(k0_, kN_ + 1)))

    s = Sum(k * x, (k, k0, kN))
    f = lambdify((k0, kN, x), s, 'numpy')

    k0_, kN_ = 0, 10
    x_ = np.linspace(-1, +1, 100)
    assert np.allclose(f(k0_, kN_, x_), sum(k_ * x_ for k_ in range(k0_, kN_ + 1)))


def test_equality():
    e = Equality(x, 1)

    f = lambdify((x,), e, 'numpy')
    x_ = np.array([0, 1, 2])
    assert np.array_equal(f(x_), [False, True, False])

    e = Unequality(x, 1)

    f = lambdify((x,), e, 'numpy')
    x_ = np.array([0, 1, 2])
    assert np.array_equal(f(x_), [True, False, True])


# test_lambdarepr.py

def test_sum():
    # In each case, test eval() the lambdarepr() to make sure that
    # it evaluates to the same results as the symbolic expression

    k, k0, kN = symbols("k, k0, kN")

    s = Sum(x ** k, (k, k0, kN))

    l = Printer().doprint(s)
    assert l == "(builtins.sum(x**k for k in range(k0, kN+1)))"

    assert (lambdify((x, k0, kN), s)(2, 3, 8) ==
            s.subs([(x, 2), (k0, 3), (kN, 8)]).doit())

    s = Sum(k * x, (k, k0, kN))

    l = Printer().doprint(s)
    assert l == "(builtins.sum(k*x for k in range(k0, kN+1)))"

    assert (lambdify((x, k0, kN), s)(2, 3, 8) ==
            s.subs([(x, 2), (k0, 3), (kN, 8)]).doit())
