# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
"""
Continuous Operators
======================

Module for symbolic computations.

>>> c = Chart(x)
>>> assert c.dim == 1
>>> c = Chart(x, y)
>>> assert c.dim == 2

>>> F0, F1 = Form.forms(x)
>>> assert F0(x) == Form(0, (x,), (x,))
>>> assert F1(x) == Form(1, (x,), (x,))

>>> F0, F1, F2 = Form.forms(x, y)
>>> assert F0(x   ) == Form(0, (x, y), (x,  ))
>>> assert F1(x, y) == Form(1, (x, y), (x, y))
>>> assert F2(x   ) == Form(2, (x, y), (x,  ))

>>> F1(x, y)
Form(1, (x, y), (x, y))
"""
from spexy.chart import Chart
from spexy.form.sym import Form
import sympy as sy

# Coordinates
x, y, z = sy.symbols('x, y, z')
# Derivatives
dx = lambda _: sy.diff(_, x)
dy = lambda _: sy.diff(_, y)
dz = lambda _: sy.diff(_, z)
# Vector Fields
u, v = sy.Function('u')(x, y), sy.Function('v')(x, y)
ux, uy = sy.Function('ux')(x, y), sy.Function('uy')(x, y)
vx, vy = sy.Function('vx')(x, y), sy.Function('vy')(x, y)
# Scalar Fields
f, g = sy.Function('f')(x, y), sy.Function('g')(x, y)
# Coordinates of Simplex Vertices
x0, y0, x1, y1, x2, y2 = sy.symbols('x0, y0, x1, y1, x2, y2')
# Form Creation Routines
F0, F1, F2 = Form.forms(x, y)


def D(f):
    """
    Derivative

    >>> assert D( F0(x) ) == F1(1, 0)
    >>> assert D( F0(x*y) ) == F1(y, x)
    >>> assert D( F1(-y, x) ) == F2(2)
    >>> assert D( F2(x) ) == 0
    """
    if f is 0:
        return 0
    return f.D


def H(f):
    """
    Hodge Star

    >>> assert H(F0(x)) == F2(x)
    >>> assert H(F1(x,y)) == F1(-y, x)
    >>> assert H(F2(x)) == F0(x)
    """
    if f is 0:
        return 0
    return f.H


def W(a, b):
    """
    Wedge Product

    >>> u, v, f, g = sy.symbols('u v f g')
    >>> assert W(F0(f),F0(g)) == F0(f*g)
    >>> assert W(F0(f),F1(u,v)) == F1(f*u, f*v)
    >>> assert W(F1(u,v),F0(f)) == F1(f*u, f*v)
    >>> assert W(F1(u,v),F1(f,g)) == F2(-f*v + g*u)
    >>> assert W(F0(f),F2(g)) == F2(f*g)
    >>> assert W(F2(g),F0(f)) == F2(f*g)
    """
    return a.W(b)


def I(a, b):
    """
    Inner Product

    The inner product between forms can be expressed as

    .. math:: \langle \alpha, \beta \rangle = \star ( \alpha \wedge \star \beta )

    >>> I(F0(f), F0(g)) == F0(f*g) == I(F0(g), F0(f))
    True
    >>> I(F1(f,g), F1(u,v)) == F0(f*u + g*v) == I(F1(u,v), F1(f,g))
    True
    >>> I(F2(f), F2(g)) == F0(f*g) == I(F2(g), F2(f))
    True
    """
    return H(W(a, H(b)))


def C(X, f):
    """
    Contraction

    >>> u, v, f, g = sy.symbols('u v f g')

    >>> X = F1(u, v)
    >>> C(X, F1(f, g)) == F0(f*u + g*v)
    True
    >>> C(X, F2(f)) == F1(-f*v, f*u)
    True
    """
    if X is 0 or f is 0:
        return 0
    return X.C(f)


def Lie(X, f):
    """
    Lie Derivative

    >>> Lie(F1(vx,vy), F0(f)) == F0( vx*dx(f) + vy*dy(f) )
    True
    >>> Lie(F1(vx,vy), F2(f)) == F2( dx(f*vx) + dy(f*vy) )
    True
    >>> ff = Lie(F1(ux,uy), F1(vx,vy))
    >>> sy.simplify(ff[0]) == ux*dx(vx) + uy*dy(vx) + dx(ux)*vx + dx(uy)*vy
    True
    >>> sy.simplify(ff[1]) == ux*dx(vy) + uy*dy(vy) + dy(ux)*vx + dy(uy)*vy
    True
    >>> ff = Lie(F1(vx,vy), F1(vx,vy))
    >>> sy.simplify(ff[0]) == sy.expand( dx((vx**2+vy**2)/2) + vx*dx(vx) + vy*dy(vx) )
    True
    >>> sy.simplify(ff[1]) == sy.expand( dy((vx**2+vy**2)/2) + vx*dx(vy) + vy*dy(vy) )
    True
    """
    return C(X, D(f)) + D(C(X, f))


def Lap(f):
    """
    Laplacian Operator

    >>> d = sy.diff
    >>> assert Lap(F0(f))   == F0( d(f,x,x) + d(f,y,y) )
    >>> assert Lap(F1(vx,vy)) == F1( d(vx,x,x) + d(vx,y,y), d(vy,x,x) + d(vy,y,y) )
    >>> assert Lap(F2(f))   == F2( d(f,x,x) + d(f,y,y) )
    """
    return H(D(H(D(f)))) + D(H(D(H(f))))


def grad(f):
    """
    Compute the gradient of a scalar field :math:`f(x,y)`.

    >>> F1(*grad(f)) == D(F0(f)) 
    True
    """
    return sy.Matrix([
        dx(f),
        dy(f),
    ])


def div(V):
    """
    Compute the divergence of a vector field :math:`V(x,y)`.

    >>> F0(div([vx, vy])) == H(D(H(F1(vx,vy))))
    True
    """
    vx, vy = V
    return dx(vx) + dy(vy)


def vort(V):
    """
    Compute the vorticity of a vector field :math:`V(x,y)`.
    
    >>> F0(vort([vx, vy])) == H(D(F1(vx,vy)))
    True
    """
    vx, vy = V
    return -dy(vx) + dx(vy)


def adv(V):
    """
    >>> adv([u, v]) + grad(u**2 + v**2)/2 == sy.simplify(sy.Matrix( Lie(F1(u,v),F1(u,v)) ))
    True
    """
    vx, vy = V
    return sy.Matrix([
        vx * dx(vx) + vy * dy(vx),
        vx * dx(vy) + vy * dy(vy),
    ])


def v_dot(V, p):
    return -adv(V) - grad(p)
