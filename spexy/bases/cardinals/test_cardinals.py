# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np

from spexy.bases.cardinals import (sym, num, nat)
from spexy.helper import func_eq
from spexy.symbolic import λ


def test_chebyshev():
    x = np.linspace(-1, +1, 21)
    for N in range(2, 4):
        for name in (
                'T', 'U', 'Uclamp', 'Tclamp',
                'dT', 'dU', 'dUclamp', 'dTclamp',
                'ddT', 'ddU', 'ddUclamp', 'ddTclamp',
        ):
            assert func_eq(
                λ(lambda x: getattr(sym, name)(N, x)),
                lambda x: getattr(num, name)(N, x),
                lambda x: getattr(nat, name)(N, x),
            )(x)


def test_roots():
    for N in range(2, 4):
        for name in (
                'xT', 'xU', 'xUclamp', 'xTclamp',
        ):
            for n in range(N):
                assert func_eq(
                    λ(lambda: getattr(sym, name)(N, n)),
                    lambda: getattr(num, name)(N, n),
                    lambda: getattr(nat, name)(N, n),
                )()


def test_vert():
    x = np.linspace(-1, +1, 21)
    for N in range(2, 4):
        for name in (
                'CT', 'CU', 'CTclamp', 'CUclamp',
        ):
            for n in range(N):
                assert func_eq(
                    λ(lambda x: getattr(sym, name)(N, n, x)),
                    lambda x: getattr(num, name)(N, n, x),
                    lambda x: getattr(nat, name)(N, n, x),
                )(x), (name, N, n, x)


def test_edge():
    x = np.linspace(-1, +1, 21)
    for N in range(2, 4):
        for name in (
                'DT', 'DU', 'DTclamp', 'DUclamp',
                'DnT', 'DnU', 'DnTclamp', 'DnUclamp',
        ):
            for n in range(N - 1):
                assert func_eq(
                    λ(lambda x: getattr(sym, name)(N, n, x)),
                    lambda x: getattr(num, name)(N, n, x),
                    lambda x: getattr(nat, name)(N, n, x),
                )(x), (name, N, n, x)
