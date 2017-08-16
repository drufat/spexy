# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import sympy as sy


def s(u, v, t):
    r = 2 + sy.sin(t) ** 2
    ρ = sy.sqrt(u ** 2 + v ** 2)
    z0 = sy.sqrt(r ** 2 - 2)
    θ = sy.atan(ρ / z0)
    φ = sy.atan2(v, u)
    x = r * sy.sin(θ) * sy.cos(φ)
    y = r * sy.sin(θ) * sy.sin(φ)
    z = r * sy.cos(θ) - z0
    return x, y, z
