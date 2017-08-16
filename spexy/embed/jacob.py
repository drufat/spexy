import numpy as np
import sympy as sy

from spexy.form.coch import Form


def SJ(s, g):
    u, v, t = sy.Dummy(), sy.Dummy(), sy.Dummy()
    s = sy.lambdify([u, v, t], s(u, v, t), 'numpy')

    ((u, v),) = g.verts()

    def _(t):
        x, y, z = s(u, v, t)
        x, y, z = np.broadcast_arrays(x, y, z)

        fx = Form(0, g, [x]).G
        fy = Form(0, g, [y]).G
        fz = Form(0, g, [z]).G

        Jxu = fx.comp[0].array[0]
        Jxv = fx.comp[1].array[0]
        Jyu = fy.comp[0].array[0]
        Jyv = fy.comp[1].array[0]
        Jzu = fz.comp[0].array[0]
        Jzv = fz.comp[1].array[0]

        return [x, y, z], [Jxu, Jxv, Jyu, Jyv, Jzu, Jzv]

    return _


def SJ_sym(s, U, V):
    u, v, t = sy.Dummy(), sy.Dummy(), sy.Dummy()
    x, y, z = s(u, v, t)

    Jxu = sy.diff(x, u)
    Jxv = sy.diff(x, v)
    Jyu = sy.diff(y, u)
    Jyv = sy.diff(y, v)
    Jzu = sy.diff(z, u)
    Jzv = sy.diff(z, v)

    S = sy.lambdify([u, v, t], [x, y, z], 'numpy')
    J = sy.lambdify([u, v, t], [Jxu, Jxv, Jyu, Jyv, Jzu, Jzv], 'numpy')

    def _(t):
        return S(U, V, t), J(U, V, t)

    return _
