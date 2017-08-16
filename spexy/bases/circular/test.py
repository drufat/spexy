# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np
import sympy as sy

from spexy.bases.circular import (sym, num, nat)
from spexy.helper import func_eq
from spexy.symbolic import λ

xs = sy.symbols('x')


def test_phi():
    x = np.linspace(0, 2 * np.pi, 21)
    for N in range(1, 4):
        assert func_eq(
            λ(lambda x: sym.phi_compact_(N, x)),
            λ(lambda x: sym.phi(N, x)),
            lambda x: num.phi(N, x),
            lambda x: nat.phi(N, x),
        )(x)
        assert func_eq(
            λ(lambda x: sym.phi_star(N, x)),
            lambda x: num.phi_star(N, x),
            lambda x: nat.phi_star(N, x),
        )(x)
        assert func_eq(
            λ(lambda x: sym.phi_grad(N, x)),
            lambda x: num.phi_grad(N, x),
            lambda x: nat.phi_grad(N, x),
        )(x)
        assert sy.simplify(sym.phi_grad(N, xs) - sy.diff(sym.phi(N, xs), xs)) == 0


def test_kappa():
    x = np.linspace(0, np.pi, 21)
    for N in range(1, 4):
        for n in range(N):
            assert func_eq(
                λ(lambda x: sym.kappa(N, n, x)),
                lambda x: num.kappa(N, n, x),
                lambda x: nat.kappa(N, n, x),
            )(x)
            assert func_eq(
                λ(lambda x: sym.kappa_star(N, n, x)),
                lambda x: num.kappa_star(N, n, x),
                lambda x: nat.kappa_star(N, n, x),
            )(x)
            assert func_eq(
                λ(lambda x: sym.kappa_grad(N, n, x)),
                lambda x: num.kappa_grad(N, n, x),
                lambda x: nat.kappa_grad(N, n, x),
            )(x)
            assert sy.simplify(sym.kappa_grad(N, n, xs) - sy.diff(sym.kappa(N, n, xs), xs)) == 0


def test_psi():
    x = np.linspace(-1, +1, 21)
    for N in range(1, 4):
        for n in range(N):
            assert func_eq(
                λ(lambda x: sym.psi(N, n, x)),
                lambda x: num.psi(N, n, x),
                lambda x: nat.psi(N, n, x),
            )(x)
            assert func_eq(
                λ(lambda x: sym.psi_star(N, n, x)),
                lambda x: num.psi_star(N, n, x),
                lambda x: nat.psi_star(N, n, x),
            )(x)
            assert func_eq(
                λ(lambda x: sym.psi_grad(N, n, x)),
                lambda x: num.psi_grad(N, n, x),
                lambda x: nat.psi_grad(N, n, x),
            )(x)
            assert sy.simplify(sym.psi_grad(N, n, xs) - sy.diff(sym.psi(N, n, xs), xs)) == 0


def test_T_U():
    x = np.linspace(-1, 1, 21)
    for N in range(5):
        assert func_eq(
            λ(lambda x: sym.T(N, x)),
            λ(lambda x: sym.T_(N, x)),
            lambda x: num.T(N, x),
        )(x)
        assert func_eq(
            λ(lambda x: sym.U(N, x)),
            λ(lambda x: sym.U_(N, x)),
            lambda x: num.U(N, x),
        )(x)
