# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from sympy import sin, cos


def V(x, y):
    return (
        -cos(x / 2) * sin(y / 2),
        +sin(x / 2) * cos(y / 2)
    )


def p(x, y):
    return -(cos(x) + cos(y)) / 4
