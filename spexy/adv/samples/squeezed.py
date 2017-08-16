# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from sympy import sin, cos


def V(x, y):
    return (
        -sin(2 * y),
        sin(x)
    )


def p(x, y):
    return -4 * cos(x) * cos(2 * y) / 5
