# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from spexy.vectorfields.transform import radial


@radial
def V(r, θ):
    return 0, 1 / r
