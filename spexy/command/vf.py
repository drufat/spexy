# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.

import matplotlib.pyplot as plt
import numpy as np

from spexy.helper import load_source
from spexy.matplot import grey_save
from spexy.symbolic import λ


def arr(source, dest):
    V = load_source(source.stem, source).V
    v = λ(lambda x, y: V(x, y))

    N = 50
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    x, y = np.meshgrid(x, y)
    vx, vy = v(x, y)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 8)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis([-1, +1, -1, +1])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.quiver(x, y, vx, vy, scale=(N + 1))
    fig.savefig(str(dest), dpi=512 / 8)


def lic(source, dest):
    V = load_source(source.stem, source).V
    V = λ(lambda x, y: V(x, y))
    N, M = 512, 512
    L = 21

    x = np.arange(N + 1)
    y = np.arange(M + 1)
    x = x[:-1] + 0.5
    y = y[:-1] + 0.5
    x, y = np.meshgrid(x, y, indexing='ij')
    T = lambda x, n: 2 * x / n - 1
    vx, vy = V(T(x, N), T(y, M))
    _, _, vx, vy = np.broadcast_arrays(x, y, vx, vy)

    from licpy.lic import runlic
    tex = runlic(vx, vy, L)
    grey_save(str(dest), tex)
