# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from spexy.symbolic import poly_simplify, replace_piecewise_default
from sympy import (
    sin, cos, tan, pi, Rational, Piecewise, poly, symbols,
    acos, Eq, S, integrate, simplify, floor, limit,
    binomial, Mod, diff, oo, series, Order
)

x = symbols('x')
N, n, k = symbols('N, n, k', integer=True)


#########################
# Chebyshev Polynomials 
#########################

def T(N, x):
    """
    Chebyshev polynomials of the first kind of order N.
    >>> T(0, x)
    1
    >>> T(1, x)
    x
    >>> T(2, x)
    2*x**2 - 1
    >>> T(3, x)
    4*x**3 - 3*x
    >>> T(4, x)
    8*x**4 - 8*x**2 + 1
    >>> T(5, x)
    16*x**5 - 20*x**3 + 5*x
    >>> T(6, x)
    32*x**6 - 48*x**4 + 18*x**2 - 1
    >>> T(7, x)
    64*x**7 - 112*x**5 + 56*x**3 - 7*x
    >>> T(8, x)
    128*x**8 - 256*x**6 + 160*x**4 - 32*x**2 + 1
    """

    if N == 0:
        return 1
    n, Tprev, T = 1, 1, x
    while n < N:
        n, Tprev, T = (n + 1), T, (2 * x * T - Tprev)
    return poly(T, x).args[0]


def U(N, x):
    """
    Chebyshev polynomials of the second kind of order N.
    >>> U(-1, x)
    0
    >>> U(0, x)
    1
    >>> U(1, x)
    2*x
    >>> U(2, x)
    4*x**2 - 1
    >>> U(3, x)
    8*x**3 - 4*x
    >>> U(4, x)
    16*x**4 - 12*x**2 + 1
    >>> U(5, x)
    32*x**5 - 32*x**3 + 6*x
    >>> U(6, x)
    64*x**6 - 80*x**4 + 24*x**2 - 1
    >>> U(7, x)
    128*x**7 - 192*x**5 + 80*x**3 - 8*x
    >>> U(8, x)
    256*x**8 - 448*x**6 + 240*x**4 - 40*x**2 + 1
    """

    if N == -1:
        return 0
    if N == 0:
        return 1
    n, Uprev, U = 1, 1, 2 * x
    while n < N:
        n, Uprev, U = (n + 1), U, (2 * x * U - Uprev)
    return poly(U, x).args[0]


def T_(N, x):
    """
    >>> [T_(n, x) - T(n, x) for n in range(10)]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """
    N = S(N)
    rslt = sum(
        binomial(N, 2 * k) * (x ** 2 - 1) ** k * x ** (N - 2 * k)
        for k in range(0, floor(N / 2) + 1)
    )
    return poly(rslt, x).args[0]


def U_(N, x):
    """
    >>> [U_(n, x) - U(n, x) for n in range(10)]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """
    N = S(N)
    rslt = sum(
        binomial(N + 1, 2 * k + 1) * (x ** 2 - 1) ** k * x ** (N - 2 * k)
        for k in range(0, floor(N / 2) + 1)
    )
    return poly(rslt, x).args[0]


def Tr(N, x):
    if N == -1:
        return 0
    if N == 0:
        return 1
    if N == 1:
        return x
    if N % 2 == 0:
        return 2 * Tr(N / 2, x) ** 2 - 1
    else:
        return 2 * Tr((N - 1) / 2, x) * Tr((N + 1) / 2, x) - x


def Ur(N, x):
    if N == -1:
        return 0
    if N == 0:
        return 1
    if N == 1:
        return 2 * x
    if N % 2 == 0:
        return 2 * Ur(N / 2, x) * Tr(N / 2, x) - 1
    else:
        return 2 * Ur((N - 1) / 2, x) * Tr((N + 1) / 2, x)


def Ts(N, x):
    return cos(N * acos(x))


def Us(N, x):
    return sin((N + 1) * acos(x)) / sin(acos(x))


def test_chebyshev_symbolic():
    def series_(N, expr):
        expr = series(expr, x, 0, N + 1)
        expr = expr.replace(Order, lambda *args: 0)
        return expr

    for N in range(1, 5):
        assert T(N, x) == T_(N, x)
        assert U(N, x) == U_(N, x)
        assert T(N, x) == poly(Tr(N, x), x).args[0]
        assert U(N, x) == poly(Ur(N, x), x).args[0]
        assert T(N, x) == series_(N, Ts(N, x))
        assert U(N, x) == series_(N, Us(N, x))


def evenodd(N, evn, odd):
    # expr = ((1 + (-1) ** N) / 2 * evn +
    #         (1 - (-1) ** N) / 2 * odd)

    # expr = Piecewise((evn, Eq((-1) ** N, 1)),
    #                  (odd, Eq((-1) ** N, -1)))

    # n = floor(N / S(2))
    # expr = Piecewise((evn, Eq(N, 2 * n)),
    #                  (odd, Eq(N, 2 * n + 1)))

    expr = Piecewise(
        (evn, Eq(Mod(N, 2), 0)),
        (odd, Eq(Mod(N, 2), 1))
    )

    return expr


########################
# Fourier Coefficients 
########################

half = Rational(1, 2)


def h(N):
    return 2 * pi / N


def coef_f(N, n):
    return Piecewise(
        (h(N), Eq(n, 0)),
        (2 * sin(n * h(N) / 2) / n, True)
    )


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
    """
     0    pi   2pi
     *----*----*
     0    1    0/2

     >>> points_periodic(2, 0)
     0
     >>> points_periodic(2, 1)
     pi
     >>> points_periodic(2, 2)
     2*pi
    """
    (N, i) = S((N, i))
    h = 2 * pi / N
    return i * h


def points_regular(N, i):
    """
     0   pi/2  pi
     *----*----*
     0    1    2

     >>> points_regular(2, 0)
     0
     >>> points_regular(2, 1)
     pi/2
     >>> points_regular(2, 2)
     pi
     >>> points_regular(N, k) == points_periodic(2*N, k).simplify()
     True
     >>> points_periodic(N, k) == points_regular(N/2, k).simplify()
     True
    """
    (N, i) = S((N, i))
    h = pi / N
    return i * h


def points_regnew(N, i):
    """
    0      pi/2     pi
    |--*----*----*--|
       0    1    2

    >>> points_regnew(2, 0)
    pi/6
    >>> points_regnew(2, 1)
    pi/2
    >>> points_regnew(2, 2)
    5*pi/6
    """
    (N, i) = S((N, i))
    return points_regular(N + 1, i + half)


def points_chebyshev(N, i):
    """
    -1    0   +1
     *----*----*
     0    1    2

    >>> points_chebyshev(2, 0)
    -1
    >>> points_chebyshev(2, 1)
    0
    >>> points_chebyshev(2, 2)
    1
    """
    (N, i) = S((N, i))
    x = points_regular(N, i)
    return -cos(x)


def points_chebnew(N, i):
    """
    -1       0      +1
     |--*----*----*--|
        0    1    2

    >>> points_chebnew(2, 0)
    -sqrt(3)/2
    >>> points_chebnew(2, 1)
    0
    >>> points_chebnew(2, 2)
    sqrt(3)/2
    """
    (N, i) = S((N, i))
    x = points_regnew(N, i)
    return -cos(x)


def clamp(xmin, xmax, x):
    """
    >>> clamp(0, 10, -1)
    0
    >>> clamp(0, 10, 3)
    3
    >>> clamp(0, 10, 100)
    10
    """
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
    r"""
    .. math::
        \varphi:&& [-1,1]\to[0,\pi]\\
                && x\mapsto\arccos(-x)

    >>> varphi(-1)
    0
    >>> varphi(+1)
    pi
    """
    return acos(-x)


def varphi_inv(x):
    r"""
    .. math::
        \varphi^{-1}:&& [0,\pi]\to[-1,1]\\
                && \theta\mapsto-\cos(\theta)

    >>> varphi_inv(0)
    -1
    >>> varphi_inv(pi)
    1
    """
    return -cos(x)


########################################
# Periodic Basis Functions
########################################

def phi_compact(N, x):
    r"""

    .. math::
        \alpha_{N}(x)=\frac{1}{N}
        \begin{cases}
            \cot\frac{x}{2}\,\sin\frac{Nx}{2} & \text{if N even,}\\
            \csc\frac{x}{2}\,\sin\frac{Nx}{2} & \text{if N odd.}
        \end{cases}

    >>> phi_compact(4, x)
    sin(2*x)/(4*tan(x/2))
    >>> phi_compact(5, x)
    sin(5*x/2)/(5*sin(x/2))
    """
    N, x = S((N, x))
    evn = (sin(N * x / 2) / tan(x / 2)) / N
    odd = (sin(N * x / 2) / sin(x / 2)) / N
    return evenodd(N, evn, odd)


def test_phi_limit():
    """
    Make sure the limit of alpha as x->0 is 1.
    """
    for N in (2, 3, 4, 5, 6):
        assert limit(phi_compact(N, x), x, 0) == 1


def phi_compact_(N, x):
    return Piecewise(
        (1, Eq(x, 0)),
        (phi_compact(N, x), True),
    )


def phi(N, x):
    """
    >>> 1*phi(1, x)
    1
    >>> 2*phi(2, x)
    cos(x) + 1
    >>> 3*phi(3, x)
    2*cos(x) + 1
    >>> 4*phi(4, x)
    2*cos(x) + cos(2*x) + 1
    >>> 5*phi(5, x)
    2*cos(x) + 2*cos(2*x) + 1
    >>> 6*phi(6, x)
    2*cos(x) + 2*cos(2*x) + cos(3*x) + 1
    >>> 7*phi(7, x)
    2*cos(x) + 2*cos(2*x) + 2*cos(3*x) + 1
    >>> 8*phi(8, x)
    2*cos(x) + 2*cos(2*x) + 2*cos(3*x) + cos(4*x) + 1

    >>> phi(7, x).subs(x, 0)
    1
    >>> phi(8, x).subs(x, 0)
    1
    """
    N, x = S((N, x))
    return sum(
        coef_a(N, k) * cos(k * x)
        for k in range(-floor(N / 2), N - floor(N / 2))
    )


def phi_grad(N, x):
    N, x = S((N, x))
    return sum(
        coef_a(N, k) * (-k * sin(k * x))
        for k in range(-floor(N / 2), N - floor(N / 2))
    )


def phi_star(N, x):
    r"""

    >>> phi_star(4, x)
    sqrt(2)*cos(x)/4 + cos(2*x)/4 + 1/(2*pi)
    >>> phi_star(5, x)
    cos(x)/(5*sqrt(-sqrt(5)/8 + 5/8)) + 2*cos(2*x)/(5*sqrt(sqrt(5)/8 + 5/8)) + 1/(2*pi)

    """
    N, x = S((N, x))
    return sum(
        coef_a_star(N, k) * cos(k * x)
        for k in range(-floor(N / 2), N - floor(N / 2)))


############################################################
# Regular Basis Functions
############################################################

def gamma(N, n):
    """

    >>> gamma(2, 0).simplify()
    0
    >>> gamma(2, 1).simplify()
    -1/2 + sqrt(2)/2
    >>> gamma(2, 2).simplify()
    0

    >>> gamma(3, 0).simplify()
    0
    >>> gamma(3, 1).simplify()
    -1/3 + sqrt(3)/3
    >>> gamma(3, 2).simplify()
    -2/3 + sqrt(3)/3
    >>> gamma(3, 3).simplify()
    0

    >>> gamma(4, 0).simplify()
    0
    >>> gamma(4, 4).simplify()
    0

    >>> gamma(5, 0).simplify()
    0
    >>> gamma(5, 5).simplify()
    0

    """
    N, n = S((N, n))
    return sum(
        sin(pi * n * m / N) * tan(pi * m / 4 / N) / 2 / N
        for m in range(-N, N)
    )


def delta(N, x):
    """
    Basis function for dual half-edge at the boundary.

    >>> simplify( -diff(cos(N*x)*(1+cos(x))/2, x) - delta(N, x))
    0
    >>> N = 3
    >>> xr = lambda i: points_regular_clamped(N, i - Rational(1,2))
    >>> delta_integrate = integrate(delta(N, x), x)
    >>> delta_  = lambda i0, i1: simplify(delta_integrate.subs(x, xr(i1)) -
    ...                                   delta_integrate.subs(x, xr(i0)))
    >>> delta_(-oo, 1)
    1
    >>> delta_(1, 2)
    0
    >>> delta_(2, 3)
    0
    >>> delta_(3, oo)
    0
    """
    N, x = S((N, x))
    # return N * (1 + cos(x)) * sin(N * x) / 2 + sin(x) * cos(N * x) / 2
    return N * sin(N * x) / 2 + (N + 1) * sin((N + 1) * x) / 4 + (N - 1) * sin((N - 1) * x) / 4


def deltapsi(N, x):
    """
    >>> N = 3
    >>> xc = lambda i: points_chebyshev_clamped(N, i - Rational(1,2))
    >>> delta_integrate = integrate(deltapsi(N, x), x)
    >>> delta_  = lambda i0, i1: simplify(delta_integrate.subs(x, xc(i1)) -
    ...                                   delta_integrate.subs(x, xc(i0)))
    >>> delta_(-oo, 1)
    1
    >>> delta_(1, 2)
    0
    >>> delta_(2, 3)
    0
    >>> delta_(3, oo)
    0

    """
    N, x = S((N, x))
    return N * U(N - 1, -x) / 2 + (N + 1) * U(N, -x) / 4 + (N - 1) * U(N - 2, -x) / 4


def correction0(N, n):
    return Piecewise(
        (Rational(1, 2), Eq(n, 0)),
        (Rational(1, 2), Eq(n, N)),
        (1, True)
    )


def correctiond1(N, n, x):
    return Piecewise(
        (delta(N, x), Eq(n, 0)),
        (delta(N, pi - x), Eq(n, N)),
        (- gamma(N, n) * delta(N, x)
         - gamma(N, N - n) * delta(N, pi - x), True)
    )


def correctionpsid1(N, n, x):
    return Piecewise(
        (deltapsi(N, x), Eq(n, 0)),
        (deltapsi(N, -x), Eq(n, N)),
        (- gamma(N, n) * deltapsi(N, x)
         - gamma(N, N - n) * deltapsi(N, -x), True)
    )


def kappa(N, n, x):
    N, n, x = S((N, n, x))
    return sum(
        coef_k(N, n, m) * cos(m * x)
        for m in range(-N, N)
    )


def kappa_grad(N, n, x):
    N, n, x = S((N, n, x))
    return sum(
        coef_k(N, n, m) * (-m * sin(m * x))
        for m in range(-N, N)
    )


def kappa_A_star(N, n, x):
    N, n, x = S((N, n, x))
    return sum(
        coef_p_star(N, n, m) * sin(m * x)
        for m in range(-N, N)
    )


def kappa_star(N, n, x):
    N, n, x = S((N, n, x))
    return sum(
        coef_k_star(N, n, m) * cos(m * x)
        for m in range(-N, N)
    )


############################################################
# Chebyshev Basis Functions
############################################################

def psi(N, n, x):
    N, n, x = S((N, n, x))
    return sum(
        coef_k(N, n, m) * T(abs(m), -x) for
        m in range(-N, N)
    )


def psi_grad(N, n, x):
    N, n, x = S((N, n, x))
    return sum(
        coef_k(N, n, m) * (-abs(m) * U(abs(m) - 1, -x)) for
        m in range(-N, N)
    )


def psi_star(N, n, x):
    N, n, x = S((N, n, x))
    return sum(
        coef_p_star(N, n, abs(m)) * U(abs(m) - 1, -x) for
        m in range(-N, N)
    )


#########################
#  Lagrange Polynomials
#########################


def lagrange_polynomial(xp, n, x):
    r"""
    Lagrange Polynomials for the set of points defined by :math:`x_m`.
    The Lagrange Polynomials are such that they are 1 at the point, and 0
    everywhere else.

    .. math::
        \psi_{n}^{0}(x)=\prod_{m=0,m\neq n}^{N-1}\frac{x-x_{m}}{x_{n}-x_{m}}

    >>> points = tuple(points_chebyshev(4, i) for i in range(5))
    >>> points
    (-1, -sqrt(2)/2, 0, sqrt(2)/2, 1)

    >>> P = lagrange_polynomial(points, 0, x)
    >>> poly_simplify(P, x)
    x**4 - x**3 - x**2/2 + x/2
    >>> [P.subs(x, xi) for xi in points]
    [1, 0, 0, 0, 0]

    >>> P = lagrange_polynomial(points, 1, x)
    >>> poly_simplify(P, x)
    -2*x**4 + sqrt(2)*x**3 + 2*x**2 - sqrt(2)*x
    >>> [P.subs(x, xi) for xi in points]
    [0, 1, 0, 0, 0]

    >>> P = lagrange_polynomial(points, 2, x)
    >>> poly_simplify(P, x)
    2*x**4 - 3*x**2 + 1
    >>> [P.subs(x, xi) for xi in points]
    [0, 0, 1, 0, 0]

    >>> P = lagrange_polynomial(points, 3, x)
    >>> poly_simplify(P, x)
    -2*x**4 - sqrt(2)*x**3 + 2*x**2 + sqrt(2)*x
    >>> [P.subs(x, xi) for xi in points]
    [0, 0, 0, 1, 0]

    >>> P = lagrange_polynomial(points, 4, x)
    >>> poly_simplify(P, x)
    x**4 + x**3 - x**2/2 - x/2
    >>> [P.subs(x, xi) for xi in points]
    [0, 0, 0, 0, 1]

    """
    rslt = 1
    for i in range(len(xp)):
        if i == n:
            continue
        rslt *= (x - xp[i]) / (xp[n] - xp[i])
    return rslt
