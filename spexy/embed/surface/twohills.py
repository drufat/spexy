# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import sympy as sy


def s(u, v, t):
    a = 2
    u, v = a * u, a * v
    x = u / a
    y = v / a
    z = (3 * u ** 2 + v ** 2) * sy.exp(1 - u ** 2 - v ** 2) / 4
    return x, y, z
