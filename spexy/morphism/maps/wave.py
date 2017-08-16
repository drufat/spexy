# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from .common import *

φ = Map(
    (s, t),
    (s, t - sin(π * s) / 3))
φ = σ(1, S(3)/4) * φ
