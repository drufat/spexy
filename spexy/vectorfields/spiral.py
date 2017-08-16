# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from spexy.vectorfields.transform import radial
from sympy import Rational


@radial
def V(r, θ, c=Rational(1, 3)):
    return c, 1 / r
