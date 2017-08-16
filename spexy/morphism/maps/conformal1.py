# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from .common import *

φ = Map((s, t), (
    s ** 2 - t ** 2,
    2 * s * t
))
φ = τ(-Rational(3, 4), -Rational(1, 2)) * σ(Rational(1, 6)) * φ * τ(2, Rational(1, 2))
