# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np
import sympy as sy

from spexy.morphism.matrix import sqrtmat, invmat

args = sy.symbols('Jxu, Jxv, Jyu, Jyv, Jzu, Jzv')
Jxu, Jxv, Jyu, Jyv, Jzu, Jzv = args
J = sy.Matrix([
    [Jxu, Jxv],
    [Jyu, Jyv],
    [Jzu, Jzv]
])

g = J.T @ J

gnumpy = sy.lambdify(args, [g[0, 0], g[0, 1], g[1, 1]], 'numpy')


def sqrt_g(*J):
    guu, guv, gvv = gnumpy(*J)
    Juu, Juv, Jvu, Jvv = sqrtmat(guu, guv, guv, gvv, module=np)
    assert np.allclose(Juv, Jvu)

    return Juu, Juv, Jvu, Jvv


def bases_id(*J):
    Jxu, Jxv, Jyu, Jyv, Jzu, Jzv = J
    _1 = np.ones_like(Jxu)
    _0 = np.zeros_like(Jxu)
    e_0 = (_1, _0)
    e_1 = (_0, _1)
    return e_0, e_1


def bases_sqrt(*J):
    Jxu, Jxv, Jyu, Jyv, Jzu, Jzv = J
    Juu, Juv, Jvu, Jvv = sqrt_g(Jxu, Jxv, Jyu, Jyv, Jzu, Jzv)
    Jinv_uu, Jinv_uv, Jinv_vu, Jinv_vv = invmat(Juu, Juv, Jvu, Jvv)
    e_0 = (Jinv_uu, Jinv_vu)
    e_1 = (Jinv_uv, Jinv_vv)
    return e_0, e_1
