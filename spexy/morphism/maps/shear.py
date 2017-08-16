# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from .common import *

θ = π / 7
φ = Map(
    (s, t),
    (cos(θ) * s, t + sin(θ) * s))
