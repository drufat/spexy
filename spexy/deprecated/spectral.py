# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np
import spexy.spectral as sp


#########
# Methods using FFT
#########

def H_exp(a):
    r"""
    :math:`\int_{x-h/2}^{x+h/2}f(\xi) \exp(i \xi) d\xi`

    This transformation is singular non-invertible.
    """

    N = len(a)
    c = sp.I_diag(N + 2, -0.5, 0.5)

    a = sp.F(a)
    b = np.roll(c[2:] * a, +1)
    return sp.Finv(b)


def H_sin(a):
    r"""
    :math:`\int_{x-h/2}^{x+h/2}f(\xi) \sin(\xi) d\xi`

    This transformation is singular non-invertible.

    >>> x = np.linspace(0, 2*np.pi, 6)[:-1]
    >>> np.round_(np.real( H_sin(np.sin(2*x)) ), 3)
    array([ 0.271,  0.438, -0.573, -0.573,  0.438])
    """

    N = len(a)
    r = (N + 2) / N

    a = sp.F(a)
    c = sp.I_diag(N + 2, -0.5 * r, 0.5 * r)

    a, c = np.fft.fftshift(a), np.fft.fftshift(c)

    b = (np.roll(c[2:] * a, +1) - np.roll(c[:-2] * a, -1)) / 2j

    b = np.fft.ifftshift(b)

    return sp.Finv(b)


def I_sin(a, b, v):
    v = sp.F(v)
    c = sp.I_diag(v.shape[0], a, b)
    v = (np.roll(c * v, +1) - np.roll(c * v, -1)) / 2j

    return sp.Finv(v)


### Chebyshev hodge-stars

def H0d_cheb(f):
    r"""

    .. math::
        \tilde{\mathbf{H}}^0 =
            {\mathbf{M}_1}^{\dagger}
             \mathbf{I}^{-\frac{h}{2}, \frac{h}{2}}
             \tilde{\mathbf{\Omega}}
             \mathbf{M}_1^+
    >>> sp.to_matrix(H0d_cheb, 2)
    array([[ 0.854,  0.146],
           [ 0.146,  0.854]])
    """

    f = sp.mirror1(f, +1)
    N = f.shape[0]
    h = 2 * np.pi / N
    f = f * sp.Omega_d(N)
    f = sp.I_space(-0.5, 0.5)(f)
    f = sp.unmirror1(f)

    return np.real(f)


def H1_cheb(f):
    r"""

    .. math::
        \mathbf{H}^1 =
            {\mathbf{M}_1}^{\dagger}
            \tilde{\mathbf{\Omega}}^{-1}
            {\mathbf{I}^{-\frac{h}{2}, \frac{h}{2}}}^{-1}
            \mathbf{M}_1^-
    >>> sp.to_matrix(H1_cheb, 2)
    array([[ 1.207, -0.207],
           [-0.207,  1.207]])
    """

    f = sp.mirror1(f, -1)
    N = f.shape[0]
    f = sp.I_space_inv(-0.5, 0.5)(f)
    f = f / sp.Omega_d(N)
    f = sp.unmirror1(f)
    return np.real(f)


def H0_cheb_alternate(f):
    r"""

    .. math::
        \mathbf{H}^0 = \mathbf{M}_0^{\dagger}
            (\mathcal{A}^{0} \mathbf{E}^{-1} \mathbf{I}^{-\frac{h}{2}, 0} +
             \mathcal{A}^{N-1} \mathbf{E}^{-1} \mathbf{I}^{0, +\frac{h}{2}})
             \mathbf{\Omega} \mathbf{E}^{1}
             \mathbf{M}_0^{+}

    >>> sp.to_matrix(H0_cheb_alternate, 3)
    array([[ 0.233,  0.077, -0.017],
           [ 0.118,  1.179,  0.118],
           [-0.017,  0.077,  0.233]])
    """
    f = sp.mirror0(f, +1)
    N = f.shape[0]
    n = 4
    f = sp.E_space(n)(f)
    s = f.shape[0] / N
    f = f * sp.Omega(f.shape[0])
    l, r = sp.I_space(-0.5 * s, 0)(f), sp.I_space(0, +0.5 * s)(f)
    l, r = sp.E_space(-n)(l), sp.E_space(-n)(r)
    l, r = sp.unmirror0(l), sp.unmirror0(r)
    l[0], r[-1] = 0, 0
    f = l + r
    f = np.real(f)
    return f


def H0_cheb(f):
    r"""
    .. math::

        \mathbf{H}^0 &=&
                \mathbf{A}
                \mathbf{M}_0^{\dagger}
                \mathbf{I}^{0, +\frac{h}{2}}
                \mathbf{\Omega}
                \mathbf{E}^{N-1}
                \mathbf{M}_0^{+}

    >>> sp.to_matrix(H0_cheb, 2)
    array([[ 0.271,  0.021],
           [ 0.229,  0.479]])
   """
    f = sp.mirror0(f, +1)
    f = sp.E_space(f.shape[0])(f)
    f = f * sp.Omega(f.shape[0])
    f = sp.I_space(0, 0.5)(f)
    f = sp.unmirror1(f)
    f = sp.matA(f)

    return np.real(f)


def H1d_cheb(f):
    r"""

    .. math::

        \tilde{\mathbf{H}}^{1} = \mathbf{M}_{0}^{\dagger}
                                     \left(\mathbf{T^{-\frac{h}{2}}}\mathbf{\Omega}^{-1}\mathbf{T}^{\frac{h}{2}}-
                                                                    \mathbf{B}\mathbf{I}^{0,\frac{h}{2}}-
                                                                    \mathbf{B}^{\dagger}\mathbf{I}^{-\frac{h}{2},0}\right)
                                     \mathbf{I}^{-\frac{h}{2},\frac{h}{2}}{}^{-1}\mathbf{M}_{0}^{-}\mathbf{C}+
                                     \mathbf{B}+\mathbf{B}^{\dagger}

    >>> sp.to_matrix(H1d_cheb, 3)
    array([[ 4.5  , -0.328,  0.5  ],
           [-0.5  ,  0.914, -0.5  ],
           [ 0.5  , -0.328,  4.5  ]])
    """

    def endpoints(f):
        f0 = sp.mirror0(sp.matC(f), -1)
        aa = f - sp.unmirror0(sp.I_space(0, 0.5)(sp.I_space_inv(-0.5, 0.5)(f0)))
        bb = f - sp.unmirror0(sp.I_space(-0.5, 0)(sp.I_space_inv(-0.5, 0.5)(f0)))
        return sp.matB(aa) + sp.matB1(bb)

    def midpoints(f):
        f = sp.mirror0(sp.matC(f), -1)
        # Shift function with S, Sinv to avoid division by zero at x=0, x=pi
        f = sp.I_space_inv(-0.5, 0.5)(f)
        f = sp.T_space(+0.5)(f)
        f = f / sp.Omega_d(f.shape[0])
        f = sp.T_space(-0.5)(f)
        f = sp.unmirror0(f)
        return f

    return midpoints(f) + endpoints(f)
