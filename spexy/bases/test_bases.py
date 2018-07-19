# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np

from spexy.bases import (periodic, regular, antiregular, chebyshev, chebnew, chebold)
from spexy.helper import func_eq
from spexy.symbolic import λ


def test_regular():
    """
    Test that κ(n, x) == φ(n, x) +/- φ(n, -x)
    """

    def _(φ, κ, x, N, sign):
        c = func_eq(
            λ(lambda n, x: φ(n)(x) + sign * φ(n)(-x)),
            λ(lambda n, x: κ(n)(x)),
        )
        for n in range(N):
            assert c(n, x)

    x = np.linspace(0, np.pi, 21)
    for N in range(2, 4):
        (φ0, φ1), (φd0, φd1) = periodic.BasesImp(2 * N, 'sym').bases()

        (κ0, κ1), (κd0, κd1) = regular.BasesImp(N, 'sym').bases(correct=False)

        _(φ0, κ0, x, N, +1)
        _(φd0, κd0, x, N, +1)

        _(φ1, κ1, x, N, +1)
        _(φd1, κd1, x, N, +1)

        (κ0, κ1), (κd0, κd1) = antiregular.BasesImp(N, 'sym').bases(correct=False)

        _(φ1, κ1, x, N, -1)
        _(φd1, κd1, x, N, -1)


def test_chebyshev():
    """
    Test that ψ(n, x) == κ_a(n, θ)/sin(θ)
    """

    from sympy import acos
    def θ(x):
        return acos(-x)

    def _(ψ, κ, x, N, W):
        c = func_eq(
            λ(lambda n, x: κ(n)(θ(x)) * W(θ(x))),
            λ(lambda n, x: ψ(n)(x)),
        )

        for n_ in range(N):
            assert c(n_, x)

    x = np.linspace(-1, +1, 21)[1:-1]
    for N in range(2, 4):
        for correct in (True, False):
            (κ0, κ1), (κd0, κd1) = antiregular.BasesImp(N, 'sym').bases(correct=correct)
            (ψ0, ψ1), (ψd0, ψd1) = chebyshev.BasesImp(N, 'sym').bases(correct=correct)

            W = lambda θ: 1
            _(ψ0, κ0, x, N, W)
            _(ψd0, κd0, x, N, W)

            from sympy import sin
            W = lambda θ: 1 / sin(θ)
            _(ψ1, κ1, x, N, W)
            _(ψd1, κd1, x, N, W)


def test_bases():
    def _(module, x):
        for dual in [0, 1]:
            for N in range(1, 4):
                for (
                        sym,
                        num,
                        nat,
                        number
                ) in zip(
                    module.BasesImp(N, 'sym').bases()[dual],
                    module.BasesImp(N, 'num').bases()[dual],
                    module.BasesImp(N, 'nat').bases()[dual],
                    module.BasesImp(N).numbers()[dual],
                ):
                    for n in range(number):
                        assert func_eq(
                            λ(sym(n)),
                            num(n),
                            nat(n),
                        )(x)

    x = np.linspace(0, 2 * np.pi, 21)
    _(periodic, x)

    x = np.linspace(0, np.pi, 21)
    _(regular, x)
    _(antiregular, x)

    x = np.linspace(-1, +1, 21)
    _(chebyshev, x)
    _(chebnew, x)
    _(chebold, x)


def test_chebold():
    x = np.linspace(-1, +1, 21)
    for N in range(2, 4):
        for dual in [0, 1]:
            for (
                    cheb1,
                    cheb2,
                    number,
            ) in zip(
                chebyshev.BasesImp(N, 'nat').bases()[dual],
                chebold.BasesImp(N, 'nat').bases()[dual],
                chebold.BasesImp(N, 'nat').numbers()[dual],
            ):
                for n in range(number):
                    assert func_eq(
                        cheb1(n),
                        cheb2(n),
                    )(x)
