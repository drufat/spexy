# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from .common import *

c = Rational(77, 100)
φ = Map((s, t), (
    s + c * ((1 - t ** 2) * (1 - s ** 2)) ** 2 * t,
    t - c * ((1 - t ** 2) * (1 - s ** 2)) ** 2 * s
))
