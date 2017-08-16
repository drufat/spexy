# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import sympy as sy


def f(x, k, x0):
    return 1 / (1 + sy.exp(-k * (x - x0)))


def g(x, k):
    return (
        f(+x, k, -sy.Rational(1, 3)) *
        f(-x, k, -sy.Rational(1, 3))
    )


def s(u, v, t):
    xmin = sy.Rational(1, 4)
    xmax = sy.Rational(1, 40)
    xk = xmin + (xmax - xmin) * (1 + sy.cos(2*t)) / 2
    k = 1 / xk
    l = lambda x: g(x, k)
    x = u
    y = v
    z = l(u) * l(v) / 2
    # z = l(sy.sqrt(u ** 2 + v ** 2)) / 2
    return x, y, z
