# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import sympy as sy


def s(u, v, t):
    x = u * (1 + sy.sin(t) / 2)
    y = v * (1 + sy.sin(t) / 2)
    z = 0
    return x, y, z
