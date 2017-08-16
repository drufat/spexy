# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from spexy.morphism.map import IdMap, Scale, Shift, Map
from sympy import sin, cos, sqrt, symbols, pi, S, Rational

# do not delete these
Map, sin, cos, sqrt, S, Rational

s, t = symbols('s, t')
δ = IdMap()
σ = Scale
τ = Shift
π = pi
