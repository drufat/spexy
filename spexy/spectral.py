# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
"""
Spectral DEC
=============
"""

import operator
from functools import reduce, wraps

import numpy as np
from numpy.testing import assert_allclose as aac

from spexy.helper import (approx, to_matrix)
from spexy.ops.num import (batch, batchall, freq, H, Hinv, S, Sinv, weave)

batch, batchall


def alpha0(N, x):
    r"""

    .. math::
        \alpha_{N}(x)=\frac{1}{N}
        \begin{cases}
            \cot\frac{x}{2}\,\sin\frac{Nx}{2} & \text{if N even,}\\
            \csc\frac{x}{2}\,\sin\frac{Nx}{2} & \text{if N odd.}
        \end{cases}


    >>> def a0(N, i): return round(alpha0(N, i*2*np.pi/N), 11)
    >>> aac([a0(5, 0), a0(5, 1), a0(5, 2)], [1, 0, 0])
    >>> aac([a0(6, 0), a0(6, 1), a0(6, 2)], [1, 0, 0])

    """
    if N % 2 == 0:
        y = (np.sin(N * x / 2) / np.tan(x / 2)) / N
    else:
        y = (np.sin(N * x / 2) / np.sin(x / 2)) / N

    if hasattr(y, '__setitem__'):
        y[x == 0] = 1
    elif x == 0:
        y = 1

    return y


def beta0(N, x):
    r"""

    .. math::
        \beta_{N}(x)=
        \begin{cases}
            \frac{1}{2\pi} -\frac{1}{4}\cos\frac{Nx}{2} +
                \frac{1}{N}\sum\limits_{n=1}^{N/2}
                    \frac{n\cos nx}{\sin\frac{n\pi}{N}} & \text{if N even,}\\
            \frac{1}{2\pi} +
                \frac{1}{N}\sum\limits_{n=1}^{(N-1)/2}
                    \frac{n\cos nx}{\sin\frac{n\pi}{N}} & \text{if N odd.}
        \end{cases}
    """
    if N % 2 == 0:
        y = 1 / (2 * np.pi) - np.cos(N * x / 2) / 4
        for n in range(1, N // 2 + 1):
            y += n * np.cos(n * x) / np.sin(n * np.pi / N) / N
    else:
        y = 1 / (2 * np.pi) + 0 * x
        for n in range(1, (N - 1) // 2 + 1):
            y += n * np.cos(n * x) / np.sin(n * np.pi / N) / N
    return y


########################################
# Mapping between semi-circle and line #
########################################

def varphi(x):
    r"""
    .. math::
        \varphi:&& [-1,1]\to[0,\pi]\\
                && x\mapsto\arccos(-x)

    >>> varphi(-1) == 0
    True
    >>> varphi(+1) == np.pi
    True
    """
    return np.arccos(-x)


def varphi_inv(x):
    r"""
    .. math::
        \varphi^{-1}:&& [0,\pi]\to[-1,1]\\
                && \theta\mapsto-\cos(\theta)

    >>> varphi_inv(0) == -1
    True
    >>> varphi_inv(np.pi) == +1
    True
    """
    return -np.cos(x)


#########################
#  Lagrange Polynomials
#########################

def lagrange_polynomials(x):
    r"""
    Lagrange Polynomials for the set of points defined by :math:`x_m`.
    The Lagrange Polynomials are such that they are 1 at the point, and 0
    everywhere else.

    .. math::
        \psi_{n}^{0}(x)=\prod_{m=0,m\neq n}^{N-1}\frac{x-x_{m}}{x_{n}-x_{m}}

    >>> L = lagrange_polynomials([0, 1, 2])
    >>> [l(0) for l in L]
    [1.0, 0.0, -0.0]
    >>> [l(1) for l in L]
    [-0.0, 1.0, 0.0]
    >>> [l(2) for l in L]
    [0.0, -0.0, 1.0]

    """
    N = len(x)
    L = [None] * N
    for i in range(N):
        def Li(y, i=i):
            gen = ((y - x[j]) / (x[i] - x[j]) for j in set(range(N)).difference([i]))
            return reduce(operator.mul, gen)

        L[i] = (Li)
    return L


########################################
# Fourirer Stuff                       #
########################################


def F(x):
    """
    >>> N = 4
    >>> Fmat = to_matrix(F, N)
    >>> Fmatinv = to_matrix(Finv, N)
    >>> aac(np.linalg.inv(Fmat), Fmatinv)
    >>> aac(Fmatinv, (1/N)*np.conj(np.transpose(Fmat)))
    """
    return np.fft.fft(x)


def Finv(x):
    """
    >>> N = 4
    >>> Fmatinv = to_matrix(Finv, N)
    """
    return np.fft.ifft(x)


def FT(x):
    """
    >>> N = 4
    >>> aac(
    ...     to_matrix(F, N).T,
    ...     to_matrix(FT, N)
    ... )
    """
    N = x.shape[0]
    return N * np.conj(Finv(np.conj(x)))


def FTinv(x):
    """
    >>> N = 4
    >>> aac(
    ...     to_matrix(Finv, N).T,
    ...     to_matrix(FTinv, N)
    ... )
    """
    N = x.shape[0]
    return np.conj(F(np.conj(x))) / N


def method_in_fourier_space(g):
    @wraps(g)
    def f(x, *args, **kwds):
        return Finv(g(F(x), *args, **kwds))

    return f


def I_diag(N, a, b):
    r"""

    The diagonal that corresponds to integration in Fourier space.
    Corresponds to :math:`f(x) \mapsto \int_{x+a}^{x+b} f(\xi) d\xi`

    .. math::
        \mathbf{I}_{\phantom{a,b}nn}^{a,b}
            =\frac{e^{inb}-e^{ina}}{in}

    .. math::
        \mathbf{I}_{\phantom{a,b}00}^{a,b}=b-a
    """
    n = freq(N)
    h = 2 * np.pi / N
    a *= h
    b *= h
    y = np.select(
        [
            n == 0,
            True
        ], [
            b - a,
            (np.exp(1j * n * b) - np.exp(1j * n * a)) / (1j * n)
        ])
    return y


def S_diag(N, a):
    r"""
    The diagonal that corresponds to shifting in Fourier Space
    Corresponds to :math:`f(x) \mapsto f(x-h)`

    .. math::
        \mathbf{S}_{\phantom{a}nn}^{a}=e^{ina}
    """
    n = freq(N)
    h = 2 * np.pi / N
    a *= h
    return np.exp(1j * n * a)


def I(f, x0, x1):
    N = f.shape[0]
    f = Finv(I_diag(N, x0, x1) * F(f))
    return np.real(f)


def Iinv(f, x0, x1):
    N = f.shape[0]
    f = Finv(F(f) / I_diag(N, x0, x1))
    return np.real(f)


def fourier_I(x, a, b):
    r"""
    Corresponds to :math:`f(x) \mapsto \int_{x+a}^{x+b} f(\xi) d\xi`
    >>> to_matrix(lambda x: fourier_I(x, -.5, .5), 3)
    array([[ 2.094,  0.   ,  0.   ],
           [ 0.   ,  1.732,  0.   ],
           [ 0.   ,  0.   ,  1.732]])
    >>> to_matrix(lambda x: fourier_I(x, -.5, .5), 4)
    array([[ 1.571,  0.   ,  0.   ,  0.   ],
           [ 0.   ,  1.414,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  1.   ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  1.414]])
    """
    N = x.shape[0]
    return x * I_diag(N, a, b)


def fourier_I_inv(x, a, b):
    """
    >>> to_matrix(lambda x: fourier_I_inv(x, -.5, .5), 3)
    array([[ 0.477,  0.   ,  0.   ],
           [ 0.   ,  0.577,  0.   ],
           [ 0.   ,  0.   ,  0.577]])
    """
    N = x.shape[0]
    return x / I_diag(N, a, b)


def fourier_S(x, a):
    r"""
    Corresponds to :math:`f(x) \mapsto f(x+a)`
    >>> to_matrix(lambda x: fourier_S(x, .5), 3)
    array([[ 1.0+0.j   ,  0.0+0.j   ,  0.0+0.j   ],
           [ 0.0+0.j   ,  0.5+0.866j,  0.0+0.j   ],
           [ 0.0+0.j   ,  0.0+0.j   ,  0.5-0.866j]])
    """
    N = x.shape[0]
    return x * S_diag(N, a)


def fourier_S_inv(x, a):
    N = x.shape[0]
    return x / S_diag(N, a)


def fourier_T(x, a):
    N = x.shape[0]
    n = freq(N)
    h = 2 * np.pi / N
    a = a * h
    return x * np.exp(1j * n * a)


def fourier_K(x, a, b):
    r"""
    Corresponds to :math:`f(x) \mapsto \int_{x+a}^{x+b} f(\xi) \sin(\xi) d\xi`
    >>> to_matrix(lambda x: fourier_K(x, 0, 1), 2)
    array([[ 0.+0.j   ,  0.-1.571j],
           [ 2.+0.j   ,  0.+0.j   ]])
    >>> to_matrix(lambda x: fourier_K(x, 0, 1), 4)
    array([[ 0.000+0.j   ,  0.000+0.785j,  0.000+0.j   ,  0.000-0.785j],
           [ 0.500-0.5j  ,  0.000+0.j   ,  0.167-0.167j,  0.000+0.j   ],
           [ 0.000+0.j   ,  0.500-0.j   ,  0.000+0.j   ,  0.500+0.j   ],
           [ 0.500+0.5j  ,  0.000+0.j   , -0.500-0.5j  ,  0.000+0.j   ]])
    """
    N = x.shape[0]
    a = a * (N + 2) / N
    b = b * (N + 2) / N

    x = np.array(x, dtype=complex)

    # Add two points
    x = _extend(x, 2)
    # Multiply by sin
    x = (np.roll(x, +1) - np.roll(x, -1)) / 2j
    # Integrate
    x *= I_diag(N + 2, a, b)
    # Remove endpoints
    x = _extend(x, -2)

    return x


def fourier_K_inv(x, a, b):
    r"""
    >>> to_matrix(lambda x: fourier_K_inv(x, 0, 1), 2)
    array([[ 0.0-0.j   ,  0.5-0.j   ],
           [ 0.0+0.637j,  0.0+0.j   ]])
    >>> to_matrix(lambda x: fourier_K_inv(x, 0, 1), 3)
    Traceback (most recent call last):
    ValueError: Singular operator.
    """
    # Make sure type is coerced to complex, otherwise numpy ignores the complex parts
    # and reverts to reals.
    x = np.array(x.copy(), dtype=complex)
    N, = x.shape
    x = np.fft.fftshift(x)

    a, b = a * (N + 2) / N, b * (N + 2) / N
    I = np.fft.fftshift(
        I_diag(N + 2, a, b)
    )
    x /= I[1:-1]

    if (np.isclose(I[0], I[N]) or
            np.isclose(I[1], I[N + 1]) or
            np.isclose(I[0] * I[1], I[N] * I[N + 1])):
        raise ValueError("Singular operator.")

    y = np.zeros(N, dtype=complex)
    E = np.sum(x[::2]);
    O = np.sum(x[1::2])
    if N % 2 == 0:
        y[0] = O / (1 - I[0] / I[N])
        y[-1] = E / (I[N + 1] / I[1] - 1)
    else:
        y[0] = (I[1] / I[N + 1] * E + O) / (1 - I[1] * I[0] / I[N] / I[N + 1])
        y[-1] = (I[N] / I[0] * E + O) / (I[N] * I[N + 1] / I[0] / I[1] - 1)
    x[0] -= y[-1] * I[N + 1] / I[1]
    x[-1] -= -y[0] * I[0] / I[N]

    x = np.hstack([[-y[0]], x, [y[-1]]])
    y[::2] = -np.cumsum(x[::2])[:-1]
    y[1::2] = np.cumsum(x[1::2][::-1])[:-1][::-1]

    y *= 2j

    y = np.fft.ifftshift(y)
    return y


def refine(x):
    """
    Resample x at a twice refined grid.
    >>> N = 4
    >>> x = np.linspace(0, 2*np.pi, N+1)[:-1]
    >>> y = np.linspace(0, 2*np.pi, 2*N+1)[:-1]
    >>> approx(refine(np.cos(x)), np.cos(y))
    True
    """
    x = weave(x, S(x))
    return x


##############################
# Discrete exterior calculus
##############################


def A_diag(N):
    r"""
    
    .. math::
        \mathbf{A}=\text{diag}\left(\begin{array}{ccccccc}\frac{1}{2} & 1 & 1 & \dots & 1 & 1 & \frac{1}{2}\end{array}\right)
        
    >>> A_diag(2)
    array([ 0.5,  0.5])
    >>> A_diag(3)
    array([ 0.5,  1. ,  0.5])

    """
    assert N > 1
    d = np.concatenate(([0.5], np.ones(N - 2), [0.5]))
    return d


def extend(f, n):
    r"""

    .. math::
        \mathbf{E}^{n}:\quad
        \begin{bmatrix}x_{0}\\
        \vdots\\
        x_{N-1}
        \end{bmatrix}
        \mapsto
        \frac{N+2n}{N}
        \begin{bmatrix}\left.\begin{array}{c}
        0\\
        \vdots\\
        0
        \end{array}\right\} n\\
        \begin{array}{c}
        x_{0}\\
        \vdots\\
        x_{N-1}
        \end{array}\\
        \left.\begin{array}{c}
        0\\
        \vdots\\
        0
        \end{array}\right\} n
        \end{bmatrix}
        
    >>> extend([1, 2, 3, 4], 4)
    array([ 2.,  4.,  0.,  0.,  0.,  0.,  6.,  8.])
    >>> _extend([1, 2, 3, 4], 2)
    array([1, 2, 0, 0, 3, 4])

    """
    f = np.array(f)
    N, = f.shape
    return ((N + n) / N) * _extend(f, n)


def _extend(f, n):
    """
    Pad or unpad with zeros at the Nyquist frequency

    Extend
    >>> _extend([1], 1)
    array([1, 0])
    >>> _extend([1, 2], 0)
    array([1, 2])
    >>> _extend([1, 2], 1)
    array([1, 0, 2])
    >>> _extend([1, 2, 3], 1)
    array([1, 2, 0, 3])
    >>> _extend([1, 2, 3, 4], 1)
    array([1, 2, 0, 3, 4])
    >>> _extend([1, 2, 3, 4], 3)
    array([1, 2, 0, 0, 0, 3, 4])

    Unextend
    >>> _extend([1, 2, 0, 0, 3, 4], -2)
    array([1, 2, 3, 4])
    >>> _extend([1, 2, 0, 0, 0, 3, 4], -3)
    array([1, 2, 3, 4])
    >>> _extend([1, 0, 3], -1)
    array([1, 3])
    >>> _extend([1, 2, 10, 20, 3, 4], -2)
    array([ 1, 22, 13,  4])
    >>> _extend([1, 2, 10, 20, 30, 3, 4], -3)
    array([ 1, 32, 13, 24])

    >>> to_matrix(lambda f: _extend(f, 1), 4)
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]])
    >>> to_matrix(lambda f: _extend(f, -1), 4)
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]])
    """
    f = np.array(f)
    N = f.shape[0]
    m = N - N // 2  # Nyquist

    g = np.zeros(N + n, dtype=f.dtype)
    g[:m] += f[:m]
    g[m + n:] += f[m:]

    return g


def mirror0(f, sign=+1):
    r"""

    .. math::
        \mathbf{M}_{0}^{\pm}:\quad\begin{bmatrix}x_{0}\\
                x_{1}\\
                \vdots\\
                x_{N-2}\\
                x_{N-1}
            \end{bmatrix}\mapsto
            \begin{bmatrix}x_{0}\\
                x_{1}\\
                \vdots\\
                x_{N-2}\\
                x_{N-1}\\
                \pm x_{N-2}\\
                \vdots\\
                \pm x_{1}
            \end{bmatrix}

    >>> mirror0(np.array([1, 2, 3]))
    array([1, 2, 3, 2])
    >>> mirror0(np.array([1, 2, 3]), -1)
    array([ 1,  2,  3, -2])
    >>> to_matrix(mirror0, 3)
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.]])
    >>> to_matrix(lambda x: mirror0(x, -1), 3)
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [-0., -1., -0.]])
   """
    return np.concatenate((f, sign * f[::-1][1:-1]))


def mirror1(f, sign=+1):
    r"""

    .. math::
        \mathbf{M}_{1}^{\pm}:\quad
        \begin{bmatrix}x_{0}\\
            x_{1}\\
            \vdots\\
            x_{N-2}\\
            x_{N-1}
        \end{bmatrix}\mapsto
        \begin{bmatrix}x_{0}\\
            x_{1}\\
            \vdots\\
            x_{N-2}\\
            x_{N-1}\\
            \pm x_{N-1}\\
            \pm x_{N-2}\\
            \vdots\\
            \pm x_{1}\\
            \pm x_{0}
        \end{bmatrix}

    >>> mirror1(np.array([1, 2, 3]))
    array([1, 2, 3, 3, 2, 1])
    >>> mirror1(np.array([1, 2, 3]), -1)
    array([ 1,  2,  3, -3, -2, -1])
    >>> to_matrix(mirror1, 2)
    array([[ 1.,  0.],
           [ 0.,  1.],
           [ 0.,  1.],
           [ 1.,  0.]])
    >>> to_matrix(lambda x: mirror1(x, -1), 2)
    array([[ 1.,  0.],
           [ 0.,  1.],
           [-0., -1.],
           [-1., -0.]])
    """
    return np.concatenate((f, sign * f[::-1]))


def unmirror0(f):
    r"""

    .. math::
        \mathbf{M}_{0}^{\dagger}:\quad\begin{bmatrix}x_{0}\\
        x_{1}\\
        \vdots\\
        x_{N-1}\\
        x_{N}\\
        \vdots\\
        x_{2N-1}
        \end{bmatrix}\mapsto\begin{bmatrix}x_{0}\\
        x_{1}\\
        \vdots\\
        x_{N-1}\\
        x_{N}\\
        x_{N+1}
        \end{bmatrix}

    >>> unmirror0(np.array([1, 2, 3, 2]))
    array([1, 2, 3])
    """
    return f[:(len(f) // 2 + 1)]


def unmirror1(f):
    r"""

    .. math::
        \mathbf{M}_{1}^{\dagger}:\quad\begin{bmatrix}x_{0}\\
        x_{1}\\
        \vdots\\
        x_{N-1}\\
        x_{N}\\
        \vdots\\
        x_{2N-1}
        \end{bmatrix}\mapsto\begin{bmatrix}x_{0}\\
        x_{1}\\
        \vdots\\
        x_{N-1}
        \end{bmatrix}

    >>> unmirror1(np.array([1, 2, 3, -3, -2, -1]))
    array([1, 2, 3])
    """
    return f[:(len(f) // 2)]


def Omega(N):
    r"""

    .. math::
        \mathbf{\Omega}_{nn}=\sin\left(nh\right)

    >>> Omega(2)
    array([ 0.,  0.])
    >>> Omega(3)
    array([ 0.   ,  0.866, -0.866])
    >>> Omega(4)
    array([ 0.,  1.,  0., -1.])
    >>> Omega(8)
    array([ 0.   ,  0.707,  1.   ,  0.707,  0.   , -0.707, -1.   , -0.707])
    """

    h = 2 * np.pi / N
    o = np.sin(np.arange(N) * h)
    return o


def Omega_d(N):
    r"""

    .. math::
        \mathbf{\tilde{\Omega}}_{nn}
            =\sin\left(\left(n+\frac{1}{2}\right)h\right)

    If :math:`\tilde{\omega}` is the length of each dual edge, then

    .. math::
        \mathbf{\tilde{\Omega}}
            =\text{diag}(\mathbf{M}_{1}^{\dagger}
            \mathcal{F}^{-1}{\mathbf{I}^{-\frac{h}{2},\frac{h}{2}}}^{-1}
            \mathcal{F}\mathbf{M}_{1}^{-}\tilde{\omega})

    >>> Omega_d(2)
    array([ 1., -1.])
    >>> Omega_d(4)
    array([ 0.707,  0.707, -0.707, -0.707])

    """

    h = 2 * np.pi / N
    o = np.sin((np.arange(N) + 0.5) * h)
    return o


def Sin(N):
    """
    >>> Sin(3)
    array([ 0.,  1.,  0.])
    """
    i = np.arange(N)
    return np.sin(i * np.pi / (N - 1))


def Sin_d(N):
    """
    >>> Sin_d(2)
    array([ 0.707,  0.707])
    """
    i = np.arange(N) + 0.5
    return np.sin(i * np.pi / N)


def half_edge_base(N):
    r"""
    This is the discrete version of :math:`\delta(x)` - the basis functions
    for the half edge at -1

    .. math::
        \mathbf{B}=
            \text{diag}\left(\underset{N}{\underbrace{
            (N-1)^{2},-\frac{1}{2},\frac{1}{2},-\frac{1}{2},\cdots}
            }\right)

    .. math::
        \mathbf{B}^\dagger=
            \text{diag}\left(\underset{N}{\underbrace{
            \cdots,-\frac{1}{2},\frac{1}{2},-\frac{1}{2},(N-1)^{2}}
            }\right)


    >>> half_edge_base(3)
    array([ 4.5, -0.5,  0.5])
    """
    a = 0.5 * np.ones(N)
    a[1::2] += -1
    a[0] += (N - 1) ** 2
    return a


def half_edge_integrals(f):
    """
    >>> from spexy.symbolic import λ
    >>> from spexy.bases.circular.num import gamma
    >>> approx( 2*half_edge_integrals(np.array([0,1,0,0])), gamma(2, 1) )
    True
    >>> approx( 2*half_edge_integrals(np.array([0,1,0,0,0,0])), gamma(3, 1) )
    True
    >>> approx( 2*half_edge_integrals(np.array([0,0,1,0,0,0])), gamma(3, 2) )
    True
    """
    f = Hinv(f)
    p = I(f, 0, 0.5)
    return p[0].real


def pick(f, n):
    r"""

    Pick the nth element in the array f.

    .. math::
        \mathcal{\mathbf{P}}^{n}:\quad\begin{bmatrix}x_{0}\\
        \vdots\\
        x_{n-1}\\
        x_{n}\\
        x_{n+1}\\
        \vdots\\
        x_{N-1}
        \end{bmatrix}\mapsto\begin{bmatrix}x_{n}\\
        \vdots\\
        x_{n}\\
        x_{n}\\
        x_{n}\\
        \vdots\\
        x_{n}
        \end{bmatrix}

    .. math::
       \mathcal{\mathbf{P}}^{0}=\begin{pmatrix}1 & 0 & 0 & 0 & 0\\
        1\\
        1\\
        1\\
        1
        \end{pmatrix},\quad\mathcal{\mathbf{P}}^{1}=\begin{pmatrix} 0 & 1 & 0 & 0 & 0\\
         & 1\\
         & 1\\
         & 1\\
         & 1
        \end{pmatrix},\dots

    """

    return f[n] * np.ones(f.shape[0])


def reverse(f):
    r"""

    Reverse array.

    .. math::
        \mathbf{R}:\quad\begin{bmatrix}x_{0}\\
        \vdots\\
        x_{n-1}\\
        x_{n}\\
        x_{n+1}\\
        \vdots\\
        x_{N-1}
        \end{bmatrix}\mapsto\begin{bmatrix}x_{N-1}\\
        \vdots\\
        x_{n+1}\\
        x_{n}\\
        x_{n-1}\\
        \vdots\\
        x_{0}
        \end{bmatrix}

    .. math::
        \mathbf{R} = \begin{pmatrix} &  &  &  & 1\\
         &  &  & 1\\
         &  & 1\\
         & 1\\
        1
        \end{pmatrix}
    """
    return f[::-1]


def S_space(a):
    return lambda f: Finv(F(f) * S_diag(f.shape[0], a))


def S_space_inv(a):
    return lambda f: Finv(F(f) / S_diag(f.shape[0], a))


def I_space(a, b):
    return lambda f: Finv(F(f) * I_diag(f.shape[0], a, b))


def I_space_inv(a, b):
    return lambda f: Finv(F(f) / I_diag(f.shape[0], a, b))


def K_space(a, b):
    return lambda f: Finv(fourier_K(F(f), a, b))


def K_space_inv(a, b):
    return lambda f: Finv(fourier_K_inv(F(f), a, b))


def T_space(a):
    return lambda f: Finv(F(f) * S_diag(f.shape[0], a))


def T_space_inv(a):
    return lambda f: Finv(F(f) / S_diag(f.shape[0], a))


def E_space(n):
    return lambda f: Finv(extend(F(f), n))


def E_space_inv(n):
    return lambda f: Finv(extend(F(f), -n))


def test_E_space():
    np.random.seed(13)
    from spexy.grid import Grid_1D

    G = Grid_1D.periodic

    for N, n in [(4, 3), (5, 3), (4, 4), (5, 4)]:
        f = G(N).rand(0)

        aac(
            f.array,
            f.R(*G(N).verts())
        )

        aac(
            E_space(n)(f.array).real,
            f.R(*G(N + n).verts())
        )


def matA(a):
    """
    >>> to_matrix(matA, 6)
    array([[ 1.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  1.]])
    """
    return np.concatenate(([a[0]], a[1:-1:2] + a[2:-1:2], [a[-1]]))


def matB(f):
    """
    >>> to_matrix(matB, 6)
    array([[ 25.5,   0. ,   0. ,   0. ,   0. ,   0. ],
           [ -0.5,  -0. ,  -0. ,  -0. ,  -0. ,  -0. ],
           [  0.5,   0. ,   0. ,   0. ,   0. ,   0. ],
           [ -0.5,  -0. ,  -0. ,  -0. ,  -0. ,  -0. ],
           [  0.5,   0. ,   0. ,   0. ,   0. ,   0. ],
           [ -0.5,  -0. ,  -0. ,  -0. ,  -0. ,  -0. ]])
    """
    b = half_edge_base(f.shape[0])
    return b * f[0]


def matB1(f):
    """
    >>> to_matrix(matB1, 6)
    array([[ -0. ,  -0. ,  -0. ,  -0. ,  -0. ,  -0.5],
           [  0. ,   0. ,   0. ,   0. ,   0. ,   0.5],
           [ -0. ,  -0. ,  -0. ,  -0. ,  -0. ,  -0.5],
           [  0. ,   0. ,   0. ,   0. ,   0. ,   0.5],
           [ -0. ,  -0. ,  -0. ,  -0. ,  -0. ,  -0.5],
           [  0. ,   0. ,   0. ,   0. ,   0. ,  25.5]])
    """
    b = half_edge_base(f.shape[0])
    return b[::-1] * f[-1]


def matC(f):
    """
    >>> to_matrix(matC, 4)
    array([[ 0.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.]])
    """
    return np.concatenate(([0], f[1:-1], [0]))


def fold1(f, sgn=+1):
    """
    >>> fold1(np.array([0, 1, 2, 3]), +1)
    array([3, 3])
    >>> fold1(np.array([0, 1, 2, 3]), -1)
    array([-3, -1])
    >>> to_matrix(fold1, 4)
    array([[ 1.,  0.,  0.,  1.],
           [ 0.,  1.,  1.,  0.]])
    >>> to_matrix(fold1, 8)
    array([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
           [ 0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.]])
    """
    return f[:f.shape[0] // 2] + sgn * f[::-1][:f.shape[0] // 2]


def fold0(f, sgn=+1):
    """
    >>> fold0(np.array([0, 1, 2, 3]), +1)
    array([0, 4, 2])
    >>> fold0(np.array([0, 1, 2, 3]), -1)
    array([ 0, -2, -2])
    >>> to_matrix(fold0, 4)
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  1.],
           [ 0.,  0.,  1.,  0.]])
    >>> to_matrix(fold0, 6)
    array([[ 1.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.,  1.],
           [ 0.,  0.,  1.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.,  0.,  0.]])
    """
    return (np.hstack([f[:f.shape[0] // 2], [0]]) +
            sgn * np.hstack([[0], f[::-1][:f.shape[0] // 2]]))


def unfold0(f):
    """
    >>> unfold0(np.array([ 1, 1, 1]))
    array([ 1. ,  0.5, -1. , -0.5])
    >>> to_matrix(unfold0, 4)
    array([[ 1. ,  0. ,  0. ,  0. ],
           [ 0. ,  0.5,  0. ,  0. ],
           [ 0. ,  0. ,  0.5,  0. ],
           [-0. , -0. , -0. , -1. ],
           [-0. , -0. , -0.5, -0. ],
           [-0. , -0.5, -0. , -0. ]])
    """
    return np.hstack([[f[0]], .5 * f[1:-1], [-f[-1]], -.5 * f[1:-1][::-1]])


def laplacian(g):
    """
        Laplacian Operator
    """
    D0, D1, D0d, D1d = g.derivative()
    H0, H1, H0d, H1d = g.hodge_star()

    L = lambda x: H1d(D0d(H1(D0(x))))
    Ld = lambda x: H1(D0(H1d(D0d(x))))

    return L, Ld
