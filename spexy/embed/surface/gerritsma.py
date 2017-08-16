import sympy as sy

c = sy.Rational(13, 100)


def s(u, v, t):
    ct = c * sy.cos(t)
    x = u + ct * sy.sin(sy.pi * u) * sy.sin(sy.pi * v)
    y = v + ct * sy.sin(sy.pi * u) * sy.sin(sy.pi * v)
    z = 0
    return x, y, z
