# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import inspect
from functools import wraps

import numpy as np
from spexy.continuous import x, y
from spexy.other.sympy import lambdify

import sympy as sy
from sympy import (solve, collect, Eq, Piecewise, symbols, sqrt, acos, pi, sympify)
from sympy.core import Function

x = symbols('x')
N, n, k = symbols('N, n, k', integer=True)


def symbolize(f):
    @wraps(f)
    def fs(*args):
        return f(*[sympify(a) for a in args])

    return fs


@symbolize
def replace_Eq(expr, x):
    """
    >>> expr = Eq(-acos(-x) + pi, 0)
    >>> replace_Eq(expr, x)
    Eq(x, 1)
    >>> expr = Eq(-acos(-x), 0)
    >>> replace_Eq(expr, x)
    Eq(x, -1)
    """

    def replace(lhs, rhs):
        if lhs != x:
            sln = solve(lhs.doit() - rhs.doit(), x)
            if len(sln) == 1:
                return Eq(x, sln[0])
        return Eq(lhs, rhs)

    return expr.replace(Eq, replace)


@symbolize
def replace_piecewise_default(expr):
    """
    Replace piecewise expression with its default value,

    >>> expr = Piecewise((1, x < 1), (x, True))
    >>> replace_piecewise_default(expr)
    x
    >>> expr = Piecewise((1, True), (2, True))
    >>> replace_piecewise_default(expr)
    1
    >>> expr = Piecewise((1, x < 1), (x, x >= 1))
    >>> replace_piecewise_default(expr)
    Piecewise((1, x < 1), (x, x >= 1))
    """

    def replace(*args):
        for expr, cond in args:
            if cond == True:
                return expr
        return Piecewise(*args)

    return expr.replace(Piecewise, replace)


def poly_simplify(p, x):
    """

    >>> poly_simplify((x-1)*(x+1), x)
    x**2 - 1
    >>> poly_simplify((x-1)/(sqrt(2)+1), x)
    x/(1 + sqrt(2)) - 1/(1 + sqrt(2))
    """
    return collect(p.expand(), x)


def λ(kappa):
    """
    >>> from spexy.bases.circular.sym import phi, points_chebyshev_clamped
    >>> phi = λ(lambda x: phi(2, x))
    >>> phi(0)
    1.0
    >>> p = λ(lambda N, i: points_chebyshev_clamped(N, i))
    >>> np.array([p(3, i) for i in range(4)])
    array([-1. , -0.5,  0.5,  1. ])
    """
    nargs = len(inspect.signature(kappa).parameters)
    dummy = tuple(sy.Dummy() for _ in range(nargs))
    return sy.lambdify(dummy, kappa(*dummy), 'numpy')


def funcify(name=None):
    """
    >>> def func(x):
    ...     return 2*x
    >>> f = funcify()(func)
    >>> f(x).doit()
    2*x
    >>> f(x, evaluate=False)
    func(x)
    >>> f = funcify('f')(func)
    >>> f(x)
    2*x
    >>> lambdify((x,), f(x).doit())(1.0)
    2.0
    """
    if name is None:
        name_ = lambda f: f.__name__
    else:
        name_ = lambda f: name

    def decorator(f):

        def eval(cls, *args):
            return f(*args)

        def doit(self, **hints):
            return f(*self.args)

        obj = {
            'eval': classmethod(eval),
            'doit': doit,
            '__doc__': f.__doc__
        }

        return type(name_(f), (Function,), obj)

    return decorator


def lambdify2():
    """
    >>> l0, l1 = lambdify2()
    >>> assert l0(x*y)(1, 2) == (lambda x, y: x*y)(1, 2)
    >>> assert l1((x, y))(1, 2) == (lambda x, y: (x,y))(1, 2)
    """

    def l0(f):
        return lambdify((x, y), f, 'numpy')

    def l1(f):
        def f_(x_, y_, f=f):
            return (lambdify((x, y), f[0], 'numpy')(x_, y_),
                    lambdify((x, y), f[1], 'numpy')(x_, y_))

        return f_

    return l0, l1