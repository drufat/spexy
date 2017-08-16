# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
######################
# Old Integration
######################
import numpy as np
from scipy.integrate import quad, dblquad

from spexy.helper import is_equidistant, approx, subdivide
from spexy.spectral import H, varphi, varphi_inv


def integrate_simpson(a, b, f):
    """
    Simpson's 3-point rule O(h**4)
    http://en.wikipedia.org/wiki/Simpson%27s_rule

    >>> integrate_simpson(0, 1, lambda x: x)
    0.5
    >>> integrate_simpson(-1, -0.5, lambda x: x)
    -0.375
    """
    I = ((b - a) / 6.0) * (f(a) + 4 * f((a + b) / 2.0) + f(b))
    return I


def integrate_boole(x1, x5, f):
    """
    Boole's 5-point rule O(h**7)
    http://en.wikipedia.org/wiki/Boole%27s_rule

    >>> integrate_boole(0, 1, lambda x: x)
    0.5
    >>> integrate_boole(0, 1, lambda x: x**4)
    0.2
    """
    h = (x5 - x1) / 4.0
    x2 = x1 + h
    x3 = x1 + 2 * h
    x4 = x1 + 3 * h
    # assert(x5 == x1 + 4*h)
    I = (2 * h / 45.0) * (7 * f(x1) + 32 * f(x2) + 12 * f(x3) + 32 * f(x4) + 7 * f(x5))
    return I


def integrate_boole1(x, f):
    """
    >>> integrate_boole1([0, 1], lambda x: x)
    array([ 0.5])
    """
    x = np.asanyarray(x)
    return integrate_boole(x[:-1], x[1:], f)


def integrate_boole2(x1, x5, f):
    """
    Boole's 5-point rule O(h**7) in 2D
    """
    h = (x5 - x1) / 4.0
    x2 = x1 + h
    x3 = x1 + 2 * h
    x4 = x1 + 3 * h
    I = (2 * np.sqrt(h[0] ** 2 + h[1] ** 2) / 45.0) * \
        (7 * f(*x1) + 32 * f(*x2) + 12 * f(*x3) + 32 * f(*x4) + 7 * f(*x5))
    return I


def integrate_quad(f, a, b):
    return np.array(tuple(quad(f, a_, b_)[0] for a_, b_ in zip(a, b)))


def integrate_1form(edge, f):
    """
    Integrate a continuous one-form **f** along an **edge** 
    ((x0, y0), (x1, y1))
    >>> integrate_1form( ((0,0), (1,0)), lambda x, y: (1, 0) )[0]
    1.0
    >>> integrate_1form( ((0,0), (1,0)), lambda x, y: (0, 1) )[0]
    0.0
    """

    def tmp(x0, y0, x1, y1):
        dx = x1 - x0
        dy = y1 - y0

        def _f(t):
            x = x0 + t * dx
            y = y0 + t * dy
            fx, fy = f(x, y)
            return fx * dx + fy * dy

        return quad(_f, 0, 1)

    ((x0, y0), (x1, y1)) = edge
    return tmp(x0, y0, x1, y1)


def _integrate_2form(face, f):
    g = lambda x: 0
    h = lambda x: 1 - x

    def tmp(x0, y0, x1, y1, x2, y2):
        dx1 = x1 - x0
        dy1 = y1 - y0
        dx2 = x2 - x0
        dy2 = y2 - y0
        _f = lambda u, v: f(x0 + u * dx1 + v * dx2,
                            y0 + u * dy1 + v * dy2) * (dx1 * dy2 - dy1 * dx2)
        return dblquad(_f, 0, 1, g, h)

    ((x0, y0), (x1, y1), (x2, y2)) = face
    return tmp(x0, y0, x1, y1, x2, y2)


def integrate_2form(face, f):
    """
    Integrate a continuous two-form **f** on a **face** 
    ((x0, y0), (x1, y1), (x2, y2), ...)
    >>> integrate_2form( ((0,0), (2,0), (0,2)), lambda x, y: (1,) )[0]
    2.0
    >>> integrate_2form( ((0,0), (1,0), (1,1), (0,1)), lambda x, y: (1,) )[0]
    1.0
    """
    integral = 0.0
    error = 0.0

    a = face[0]
    for b, c in zip(face[1:-1], face[2:]):
        i, e = _integrate_2form((a, b, c), lambda *args: f(*args)[0])
        integral += i
        error += e

    return integral, error


def integrate_spectral_coarse(x, f):
    """
    >>> integrate_spectral_coarse(np.linspace(-np.pi, np.pi, 3), np.sin)
    array([-2.,  2.])
    """
    assert is_equidistant(x)
    assert approx(2 * np.pi, x[-1] - x[0]), x
    f0 = f(x[:-1] + 0.5 * np.diff(x))
    f1 = np.real(H(f0))
    return f1


def integrate_spectral(x, f):
    """
    >>> integrate_spectral(np.linspace(-np.pi, np.pi, 3), np.sin)
    array([-2.,  2.])
    """
    assert is_equidistant(x)

    r = subdivide
    f1 = integrate_spectral_coarse(r(r(r(x))), f)

    return (f1[0::8] + f1[1::8] + f1[2::8] + f1[3::8] +
            f1[4::8] + f1[5::8] + f1[6::8] + f1[7::8])


def integrate_chebyshev(xi, f):
    """
    >>> integrate_chebyshev(np.array([np.cos(np.pi), np.cos(.5*np.pi), np.cos(0)]), lambda x: x)
    array([-0.5,  0.5])
    """
    assert approx(2, xi[-1] - xi[0]), xi

    F = lambda theta: f(-np.cos(theta)) * np.sin(theta)
    x = varphi(xi)

    assert (is_equidistant(x))

    # complete the circle from the other side
    x = np.concatenate((x, x[1:] + np.pi))
    return integrate_spectral(x, F)[:len(xi) - 1]


def integrate_chebyshev_dual(xi, f):
    """
    Integrate points that may include the two half-edges at the boundary.
    #>>> integrate_chebyshev_dual(array([cos(pi), cos(0.75*pi), cos(0.25*pi), cos(0)]), lambda x: x)
    #array([-0.25,  0.  ,  0.25])
    """
    x = varphi(xi)
    z = varphi_inv(np.concatenate(([x[0]], subdivide(x[1:-1]), [x[-1]])))
    i, j = np.concatenate(([0], integrate_chebyshev(z, f), [0])).reshape(-1, 2).T
    return i + j


def split_args(I):
    """
    Convert integration function from I(x, f) to I(x0, x1, f) form
    """

    def J(x0, x1, f, I=I):
        from numpy.testing import assert_almost_equal
        assert_almost_equal(x0[1:], x1[:-1])
        return I(np.concatenate((x0, [x1[-1]])), f)

    return J
