# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from .common import *

c = Rational(13, 100)
φ = Map((s, t), (
    s + c * sin(π * s) * sin(π * t),
    t + c * sin(π * s) * sin(π * t)
))
