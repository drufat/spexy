# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from .common import *

S = σ(Rational(1, 2)) * τ(1, 1)
φ = Map(
    (s, t),
    (s ** 3, t ** 3))
φ = S.inv() * φ * S
