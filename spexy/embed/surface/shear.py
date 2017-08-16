# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import sympy as sy

def s(u, v, t):
    x = u + v * sy.cos(t) / 2
    y = v
    z = 0
    return x, y, z
