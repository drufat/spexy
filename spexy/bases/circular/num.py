# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from math import floor
import numpy as np
from numpy import (arccos, sin, cos, tan, pi, select, array)

acos = np.arccos
S = lambda _: _
half = 0.5


def Piecewise(*args): return np.select(*tuple(zip(*args))[::-1])


#########################
# Chebyshev Polynomials 
#########################

def T(n, x):
    theta = acos(x)
    return cos(n * theta)


def U(n, x):
    theta = acos(x)
    return Piecewise(
        (n + 1, x >= +1.0),
        ((n + 1) * (-1) ** n, x <= -1.0),
        (sin((n + 1) * theta) / sin(theta), True)
    )


########################
# Fourier Coefficients 
########################


def h(N):
    return 2 * pi / N


def coef_f(N, n):
    if n == 0:
        return h(N)
    else:
        return 2 * sin(n * h(N) / 2) / n


def coef_a(N, n):
    return 1 / N


def coef_a_star(N, n):
    return coef_a(N, n) / coef_f(N, n)


def coef_k(N, n, m):
    return coef_a(2 * N, m) * 2 * cos(pi * m * n / N)


def coef_k_star(N, n, m):
    return coef_a_star(2 * N, m) * 2 * cos(pi * m * n / N)


def coef_p_star(N, n, m):
    return coef_a_star(2 * N, m) * 2 * sin(pi * m * n / N)


############
# Points
############

def points_periodic(N, i):
    h = 2 * pi / N
    return i * h


def points_regular(N, i):
    h = pi / N
    return i * h


def points_regnew(N, i):
    return points_regular(N + 1, i + half)


def points_chebyshev(N, i):
    x = points_regular(N, i)
    return -cos(x)


def points_chebnew(N, i):
    x = points_regnew(N, i)
    return -cos(x)


def clamp(xmin, xmax, x):
    xmin, xmax, x = S((xmin, xmax, x))
    return Piecewise(
        (xmin, x < xmin),
        (xmax, x > xmax),
        (x, True)
    )


def points_regular_clamped(N, i):
    return points_regular(N, clamp(0, N, i))


def points_chebyshev_clamped(N, i):
    return points_chebyshev(N, clamp(0, N, i))


########################################
# Mapping between semi-circle and line #
########################################

def varphi(x):
    return arccos(-x)


def varphi_inv(x):
    return -cos(x)


########################################
# Periodic Basis Functions
########################################

def phi_compact(N, x):
    if N % 2 == 0:
        return (sin(N * x / 2) / tan(x / 2)) / N
    else:
        return (sin(N * x / 2) / sin(x / 2)) / N


def phi(N, x):
    return sum(
        coef_a(N, k) * cos(k * x)
        for k in range(-floor(N / 2), N - floor(N / 2))
    )


def phi_grad(N, x):
    return sum(
        coef_a(N, k) * (-k * sin(k * x))
        for k in range(-floor(N / 2), N - floor(N / 2))
    )


def phi_star(N, x):
    return sum(
        coef_a_star(N, k) * cos(k * x)
        for k in range(-floor(N / 2), N - floor(N / 2)))


############################################################
# Regular Basis Functions
############################################################

def gamma(N, n):
    return sum(
        sin(pi * n * m / N) * tan(pi * m / 4 / N) / 2 / N
        for m in range(-N, N)
    )


def delta(N, x):
    # return N * (1 + cos(x)) * sin(N * x) / 2 + sin(x) * cos(N * x) / 2
    return N * sin(N * x) / 2 + (N + 1) * sin((N + 1) * x) / 4 + (N - 1) * sin((N - 1) * x) / 4


def deltapsi(N, x):
    return N * U(N - 1, -x) / 2 + (N + 1) * U(N, -x) / 4 + (N - 1) * U(N - 2, -x) / 4


def correction0(N, n):
    if n == 0:
        return 1 / 2
    elif n == N:
        return 1 / 2
    else:
        return 1


def correctiond1(N, n, x):
    if n == 0:
        return delta(N, x)
    elif n == N:
        return delta(N, pi - x)
    else:
        return (- gamma(N, n) * delta(N, x)
                - gamma(N, N - n) * delta(N, pi - x))


def correctionpsid1(N, n, x):
    if n == 0:
        return deltapsi(N, x)
    elif n == N:
        return deltapsi(N, -x)
    else:
        return (- gamma(N, n) * deltapsi(N, x)
                - gamma(N, N - n) * deltapsi(N, -x))


def kappa(N, n, x):
    n, x = array(n), array(x)
    return sum(
        coef_k(N, n, m) * cos(m * x)
        for m in range(-N, N)
    )


def kappa_grad(N, n, x):
    n, x = array(n), array(x)
    return sum(
        coef_k(N, n, m) * (-m * sin(m * x))
        for m in range(-N, N)
    )


def kappa_A_star(N, n, x):
    n, x = array(n), array(x)
    return sum(
        coef_p_star(N, n, m) * sin(m * x)
        for m in range(-N, N)
    )


def kappa_star(N, n, x):
    n, x = array(n), array(x)
    return sum(
        coef_k_star(N, n, m) * cos(m * x)
        for m in range(-N, N)
    )


############################################################
# Chebyshev Basis Functions
############################################################

def psi(N, n, x):
    n, x = array(n), array(x)
    return sum(
        coef_k(N, n, m) * T(abs(m), -x) for
        m in range(-N, N)
    )


def psi_grad(N, n, x):
    n, x = array(n), array(x)
    return sum(
        coef_k(N, n, m) * (-abs(m) * U(abs(m) - 1, -x)) for
        m in range(-N, N)
    )


def psi_star(N, n, x):
    n, x = array(n), array(x)
    return sum(
        coef_p_star(N, n, abs(m)) * U(abs(m) - 1, -x) for
        m in range(-N, N)
    )
