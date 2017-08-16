# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import sympy as sy


def s(u, v, t):
    w = sy.sin(t)
    x = u * sy.sqrt(1 - v ** 2 / 2 - w ** 2 / 2 + v ** 2 * w ** 2 / 3)
    y = v * sy.sqrt(1 - w ** 2 / 2 - u ** 2 / 2 + w ** 2 * u ** 2 / 3)
    z = w * sy.sqrt(1 - u ** 2 / 2 - v ** 2 / 2 + u ** 2 * v ** 2 / 3)

    s = sy.S(3) / 2
    x, y, z = s * x, s * y, s * (z - w)
    return x, y, z
