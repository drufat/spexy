# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import sympy as sy


def s(u, v, t):
    x = u
    y = v
    xt = 2 * x + sy.cos(t) / 2
    yt = 2 * y + sy.sin(t) / 2
    z = sy.exp(-xt ** 2 - yt ** 2)
    return x, y, z
