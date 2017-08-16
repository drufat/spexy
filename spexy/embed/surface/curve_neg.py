# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import sympy as sy


def s(u, v, t):
    x = u
    y = v
    z = (x ** 2 - y ** 2) * sy.cos(t)
    return x, y, z
