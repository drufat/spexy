# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np


def coordinate_delta(deltax, deltay, shape):
    if deltax is None:
        deltax = np.ones(shape[0] - 1)
    if deltay is None:
        deltay = np.ones(shape[1] - 1)
    u = np.concatenate([[0], np.cumsum(deltax)]) / np.sum(deltax)
    v = np.concatenate([[0], np.cumsum(deltay)]) / np.sum(deltay)
    return u, v


def surface_indices(n, m):
    indices = np.arange(n * m).reshape(n, m)
    i0 = indices[:-1, :-1]
    i1 = indices[+1:, :-1]
    i2 = indices[+1:, +1:]
    i3 = indices[:-1, +1:]
    indices = [i0, i1, i2, i2, i3, i0]
    indices = np.rollaxis(np.array(indices), 0, 3)
    return indices


def boundary_indices(n, m):
    indices = np.concatenate([
        np.arange(1, m),
        np.arange(1, n) * m + m - 1,
        n * m - 1 - np.arange(1, m),
        (n - 1) * m - m * np.arange(1, n),
    ])
    return indices


def cartesian(u, v):
    U, V = np.meshgrid(u, v, indexing='ij')

    N, M = U.shape
    s_indices = surface_indices(N, M)
    b_indices = boundary_indices(N, M)

    return U, V, s_indices, b_indices
