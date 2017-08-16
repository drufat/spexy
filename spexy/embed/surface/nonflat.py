# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import sympy as sy


def s(u, v, t):
    u, v = 3 * u, 3 * v
    x = u / 3
    y = v / 3
    z = u * v * (u ** 2 - v ** 2) / (u ** 2 + v ** 2) / 9
    return x, y, z
