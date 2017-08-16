# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import sympy as sy


def s(u, v, t):
    a = sy.Rational(5, 2)
    u, v = a * u, a * v
    x = u / a
    y = v / a
    z = (u ** 2 + v ** 2) * sy.exp(-u ** 2 - v ** 2) * 2
    return x, y, z
