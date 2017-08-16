# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from sympy import sin, cos


def V(x, y):
    return (
        -2 * sin(y) * cos(x / 2) ** 2,
        +2 * sin(x) * cos(y / 2) ** 2,
    )


def p(x, y):
    return (
               - cos(2 * x) * (5 + 4 * cos(y))
               - 5 * (4 * cos(y) + cos(2 * y))
               - 4 * cos(x) * (5 + 5 * cos(y) + cos(2 * y))
           ) / 20
