# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import sympy as sy

# Coordinates
x, y, z = sy.symbols('x, y, z')
# Coordinates of Simplex Vertices
x0, y0, x1, y1, x2, y2 = sy.symbols('x0, y0, x1, y1, x2, y2')


class Chart(object):
    def __init__(self, *coord):
        dim = len(coord)
        if len(coord) == 1:
            x, = coord
            D = derivative_1d(x)
            H = hodge_star_1d()
            W = wedge_1d()
            C = contraction_1d()
        elif len(coord) == 2:
            x, y = coord
            D = derivative_2d(x, y)
            H = hodge_star_2d()
            W = wedge_2d()
            C = contraction_2d()
        else:
            raise NotImplementedError

        self.coord = coord
        self.dim = dim
        self.D = D
        self.W = W
        self.H = H
        self.C = C

    def cell_coords(self, deg):
        assert deg <= self.dim
        return enumerate_vertices(self.coord, deg)

    def __repr__(self):
        return "Chart{}".format(self.coord)

    def __eq__(self, other):
        return (
            self.coord == other.coord and
            self.dim == other.dim
        )


def Chart_1d(x=x):
    """
    >>> Chart_1d()
    Chart(x,)
    """
    return Chart(x, )


def Chart_2d(x=x, y=y):
    """
    >>> Chart_2d()
    Chart(x, y)
    """
    return Chart(x, y)


def enumerate_vertices(coord, deg):
    """
    >>> x, y = sy.symbols('x, y')
    >>> enumerate_vertices((x,), 0)
    (x0,)
    >>> enumerate_vertices((x,), 1)
    (x0, x1)
    >>> enumerate_vertices((x,), 2)
    (x0, x1, x2)
    >>> enumerate_vertices((x, y), 2)
    (x0, y0, x1, y1, x2, y2)
    """
    return sy.sympify(tuple('{}{}'.format(c.name, i) for i in range(deg + 1) for c in coord))


def simplex_measure(σ):
    n, k = len(σ[0]), len(σ) - 1
    assert all(n == len(s) for s in σ)

    if n == 1:
        if k == 0:
            return 1
        if k == 1:
            ((x0,), (x1,)) = σ
            return x1 - x0

    if n == 2:
        if k == 0:
            return 1
        if k == 1:
            ((x0, y0), (x1, y1)) = σ
            return sy.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        if k == 2:
            ((x0, y0), (x1, y1), (x2, y2)) = σ
            return ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / 2

    if n == 3:
        if k == 0:
            return 1
        if k == 1:
            ((x0, y0, z0), (x1, y1, z1)) = σ
            return sy.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        if k == 2:
            ((x0, y0, z0), (x1, y1, z1), (x2, y2, z2)) = σ
            raise NotImplementedError
        if k == 3:
            ((x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3)) = σ
            raise NotImplementedError


def derivative_1d(x):
    """
    >>> D = derivative_1d(x)
    >>> D[0]( (x,) )
    (1,)
    >>> D[0]( (x*y,) )
    (y,)
    >>> D[1]( (x,) )
    0
    """
    D0 = lambda f: (sy.diff(f[0], x),)
    D1 = lambda f: 0
    return D0, D1


def derivative_2d(x, y):
    """
    >>> D = derivative_2d(x, y)
    >>> D[0]( (x,) )
    (1, 0)
    >>> D[0]( (x*y,) )
    (y, x)
    >>> D[1]( (-y, x) )
    (2,)
    >>> D[2]( (x,) )
    0
    """
    Dx = lambda f: sy.diff(f, x)
    Dy = lambda f: sy.diff(f, y)
    D0 = lambda f: (Dx(f[0]), Dy(f[0]),)
    D1 = lambda f: (-Dy(f[0]) + Dx(f[1]),)
    D2 = lambda f: 0
    return D0, D1, D2


def hodge_star_1d():
    """
    >>> H = hodge_star_1d()
    >>> H[0]((x,))
    (x,)
    >>> H[1]((x,))
    (x,)
    """
    H0 = lambda f: f
    H1 = lambda f: f
    return H0, H1


def hodge_star_2d():
    """
    >>> H = hodge_star_2d()
    >>> H[0]((x,))
    (x,)
    >>> H[1]((x,y))
    (-y, x)
    >>> H[2]((x,))
    (x,)
    """
    H0 = lambda f: f
    H1 = lambda f: (-f[1], f[0])
    H2 = lambda f: f
    return H0, H1, H2


def antisymmetrize_wedge(W):
    r"""
    :math:`\alpha\wedge\beta` is **anticommutative**:
    :math:`\alpha\wedge\beta=(-1)^{kl}\beta\wedge\alpha`, where
    :math:`\alpha` is a :math:`k`-Form and :math:`\beta` is an
    :math:`l`-Form.
    """
    keys = [key for key in W]
    for k, l in keys:
        if k == l: continue
        W[l, k] = (lambda k, l: (
            lambda a, b:
            tuple((-1) ** (k * l) * c for c in W[k, l](b, a))
        ))(k, l)


def wedge_1d():
    """
    >>> u, v, f, g = sy.symbols('u v f g')
    >>> W = wedge_1d()
    >>> W[0,0]((f,),(g,))
    (f*g,)
    >>> W[0,1]((f,),(u,))
    (f*u,)
    >>> W[1,0]((u,),(f,))
    (f*u,)
    """
    W = {}
    W[0, 0] = lambda a, b: (a[0] * b[0],)
    W[0, 1] = lambda a, b: (a[0] * b[0],)
    antisymmetrize_wedge(W)
    return W


def wedge_2d():
    """
    >>> u, v, f, g = sy.symbols('u v f g')
    >>> W = wedge_2d()
    >>> W[0,0]((f,),(g,))
    (f*g,)
    >>> W[0,1]((f,),(u,v))
    (f*u, f*v)
    >>> W[1,0]((u,v),(f,))
    (f*u, f*v)
    >>> W[1,1]((u,v),(f,g))
    (-f*v + g*u,)
    >>> W[0,2]((f,),(g,))
    (f*g,)
    >>> W[2,0]((g,),(f,))
    (f*g,)
    """
    W = {}
    W[0, 0] = lambda a, b: (a[0] * b[0],)
    W[0, 1] = lambda a, b: (a[0] * b[0], a[0] * b[1])
    W[0, 2] = lambda a, b: (a[0] * b[0],)
    W[1, 1] = lambda a, b: (a[0] * b[1] - a[1] * b[0],)
    antisymmetrize_wedge(W)
    return W


def contraction_1d():
    """
    Contraction
    >>> C = contraction_1d()
    >>> u, v, f, g = sy.symbols('u v f g')
    >>> X = (u,)
    >>> C[0](X, (f,))
    0
    >>> C[1](X, (f,))
    (f*u,)
    """
    C0 = lambda X, f: 0
    C1 = lambda X, f: (X[0] * f[0],)
    return C0, C1


def contraction_2d():
    """
    Contraction
    >>> C = contraction_2d()
    >>> u, v, f, g = sy.symbols('u v f g')
    >>> X = (u, v)
    >>> C[0](X, (f,))
    0
    >>> C[1](X, (f,g))
    (f*u + g*v,)
    >>> C[2](X, (f,))
    (-f*v, f*u)
    """
    C0 = lambda X, f: 0
    C1 = lambda X, f: (X[0] * f[0] + X[1] * f[1],)
    C2 = lambda X, f: (-X[1] * f[0],
                       X[0] * f[0],)
    return C0, C1, C2
