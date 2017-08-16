# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from sympy import sin, cos


def V(x, y):
    return (-sin(y), sin(x))


def p(x, y):
    return -cos(x) * cos(y)
