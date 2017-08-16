# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from .common import *

φ = Map((s, t), (
    s ** 2 - t ** 2,
    2 * s * t
))
φ = τ(-1, 0) * σ(Rational(1, 6)) * φ * τ(2, 0)
