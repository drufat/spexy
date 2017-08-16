# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import sympy as sy


def s(u, v, t):
    x = u * sy.sqrt(1 - v ** 2 / 2)
    y = v * sy.sqrt(1 - u ** 2 / 2)
    z = 0
    return x, y, z
