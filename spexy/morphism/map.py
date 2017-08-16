# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.

import sympy as sy


class Map(object):
    """
    >>> s, t, x, y = sy.symbols('s, t, x, y')
    >>> φ = Map((s, t), (x(s, t), y(s, t)))
    >>> φ.J
    Matrix([
    [Derivative(x(s, t), s), Derivative(x(s, t), t)],
    [Derivative(y(s, t), s), Derivative(y(s, t), t)]])
    >>> φ = Map((x, y), (-y, x))
    >>> φ
    Map((x, y), (-y, x))
    >>> φ(s, t)
    (-t, s)
    >>> φ * φ
    Map((x, y), (-x, -y))
    >>> φ.inv()
    Map((x, y), (y, -x))
    >>> φ.jacobian()
    Matrix([
    [0, -1],
    [1,  0]])
    >>> φ.metric()
    Matrix([
    [1, 0],
    [0, 1]])
    >>> φ.metric_invariants()
    (2, 1, 1)
    >>> print(φ.ccode())
    y0 = -x1
    y1 = x0
    <BLANKLINE>
    """

    def __init__(self, xx, yy):
        assert tuple(type(_) for _ in xx) == (sy.Symbol, sy.Symbol)

        J = sy.Matrix([[sy.diff(y, x) for x in xx] for y in yy])
        g = J.T * J

        self.xx = xx
        self.yy = yy
        self.J = J
        self.g = g

    def __call__(self, *tt):
        x2t = tuple(zip(self.xx, tt))
        return tuple(y.subs(x2t, simultaneous=True) for y in self.yy)

    def __mul__(self, other):
        assert type(other) is Map
        return Map(other.xx, self(*other.yy))

    def __repr__(self):
        tup = (self.xx, self.yy)
        r = 'Map{}'.format(tup.__repr__())
        return r

    def inv(self):
        assert len(self.xx) == len(self.yy)
        tt = tuple(sy.Dummy() for _ in range(len(self.yy)))
        yt = tuple(zip(self.yy, tt))
        sln = sy.solve([y - t for (y, t) in yt], self.xx)
        if type(sln) is not dict:
            raise RuntimeError("Unable to invert map {map}.".format(map=self))
        t2x = tuple(zip(tt, self.xx))
        yyinv = tuple([sln[x].subs(t2x, simulateneous=True) for x in self.xx])
        return Map(self.xx, yyinv)

    def lambdify(self):
        return sy.lambdify(self.xx, self.yy, 'numpy')

    def ccode(self):
        xx = sy.symbols(['x{}'.format(i) for i in range(len(self.xx))])
        yy = self(*xx)
        code = ''
        for (i, y) in enumerate(yy):
            code += 'y{} = {}\n'.format(i, sy.printing.ccode(y))

        return code

    def jacobian(self):
        return self.J

    def metric(self):
        return self.g

    def metric_invariants(self):
        g = self.metric()

        I = g.trace()
        II = (g.trace() ** 2 - (g * g).trace()) / 2
        III = g.det()

        return I, II, III


def Scale(*args):
    s, t = sy.symbols('s, t')
    if len(args) == 1:
        f, = args
        return Map((s, t), (f * s, f * t))
    elif len(args) == 2:
        fx, fy = args
        return Map((s, t), (fx * s, fy * t))
    else:
        raise TypeError('Invalid arguments to scale: {}'.format(args))


def Shift(x=0, y=0):
    s, t = sy.symbols('s, t')
    return Map((s, t), (s + x, t + y))


def IdMap():
    """
    The identity map.
    """
    s, t = sy.symbols('s, t')
    return Map((s, t), (s, t))
