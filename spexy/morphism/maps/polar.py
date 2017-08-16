# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from .common import *

φ = Map(
    (s, t),
    (s * cos(t), s * sin(t)))
φ = σ(Rational(39, 100)) * τ(-2, 0) * φ * τ(2, 0)
