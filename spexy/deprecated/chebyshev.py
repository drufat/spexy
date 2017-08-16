# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import spexy.spectral as sp
import numpy as np


def H0d(f):
    """
    >>> sp.to_matrix(H0d, 2)
    array([[ 0.854,  0.146],
           [ 0.146,  0.854]])
    """
    f = sp.mirror1(f, +1)
    f = sp.F(f)
    f = sp.fourier_S(f, -0.5)
    f = sp.fourier_K(f, 0, 1)
    f = sp.Finv(f)
    f = sp.unmirror1(f)
    return np.real(f)


def H1(f):
    """
    >>> sp.to_matrix(H1, 2)
    array([[ 1.207, -0.207],
           [-0.207,  1.207]])
    """
    f = sp.mirror1(f, -1)
    f = sp.F(f)
    f = sp.fourier_K_inv(f, 0, 1)
    f = sp.fourier_S(f, +0.5)
    f = sp.Finv(f)
    f = sp.unmirror1(f)
    return np.real(f)


def H0(f):
    """
    >>> sp.to_matrix(H0, 2)
    array([[ 0.75,  0.25],
           [ 0.25,  0.75]])
    >>> sp.to_matrix(H0, 3)
    array([[ 0.233,  0.077, -0.017],
           [ 0.118,  1.179,  0.118],
           [-0.017,  0.077,  0.233]])
    """
    f = sp.mirror0(f, +1)
    f = sp.F(f)
    f = sp.fourier_K(f, 0, 0.5)
    f = sp.Finv(f)
    f = sp.fold0(f, -1)
    return np.real(f)


def H1d(f):
    """
    >>> sp.to_matrix(H1d, 2)
    array([[ 1.5, -0.5],
           [-0.5,  1.5]])
    >>> sp.to_matrix(H1d, 3)
    array([[ 4.5  , -0.328,  0.5  ],
           [-0.5  ,  0.914, -0.5  ],
           [ 0.5  , -0.328,  4.5  ]])
    """

    # Is this essentially Schur's complement? Yes, it is.
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

    f = midpoints(f) + endpoints(f)
    return np.real(f)
