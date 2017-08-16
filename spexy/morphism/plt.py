# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np
import sympy as sy
from matplotlib import collections as mc
from matplotlib import pyplot as plt

from spexy.morphism.map import IdMap
from spexy.morphism.matrix import (matrix, matrixlambdify, matrixnum, rotation, sqrtmat, invmat, normalize)


def plot_st(ax, st, expr, cmap=plt.cm.seismic, title=None, center=None, colorbar=False):
    """
    Plot a symbolic expression
    """
    X = Y = np.linspace(-1, +1, 1000)
    X, Y = np.meshgrid(X, Y)
    s, t = st
    Z = sy.lambdify((s, t), (expr,), 'numpy')(X, Y)[0] + 0 * X
    kwargs = {}
    if center is not None:
        zmin, zmax = Z.min(), Z.max()
        d = max(center - zmin, zmax - center)
        if np.allclose(d, 0):
            d += 1
        vmin = center - d
        vmax = center + d
        kwargs['vmin'], kwargs['vmax'] = vmin, vmax
    con = ax.contourf(X, Y, Z, cmap=cmap, **kwargs)
    if colorbar:
        plt.colorbar(con)
    if title is not None:
        ax.set_title(title)
    ax.axes.set_aspect('equal')
    ax.axis('off')


def plot_grid(ax, n=16, m=16, sub=10, φ=IdMap(), **kwargs):
    xmin, xmax = -1, +1
    ymin, ymax = -1, +1

    ax.set_xlim(xmin - .01, xmax + .01)
    ax.set_ylim(ymin - .01, ymax + .01)
    ax.axes.set_aspect('equal')
    ax.axis('off')

    nn = sub * n
    mm = sub * m
    x = np.linspace(xmin, xmax, n)
    y = np.linspace(ymin, ymax, m)
    xx = np.linspace(xmin, xmax, nn)
    yy = np.linspace(ymin, ymax, mm)

    xyy = np.vstack([np.repeat(x, mm), np.tile(yy, n)]).T.reshape(n, mm, 2)
    xxy = np.vstack([np.tile(xx, m), np.repeat(y, nn)]).T.reshape(m, nn, 2)

    def lambdify_mapping(φ):
        s, t = φ.xx
        xd, yd = φ.yy
        c = sy.symbols('c')
        newmap = ((1 - c) * s + c * xd, (1 - c) * t + c * yd)

        varphi = sy.lambdify((s, t, c), newmap, 'numpy')
        return varphi

    varphi = lambdify_mapping(φ)

    def forward(xy, c):
        ret = xy.copy()
        xl, yl = xy[:, :, 0], xy[:, :, 1]
        ret[:, :, 0], ret[:, :, 1] = varphi(xl, yl, c)
        return ret

    lc_xxy = mc.LineCollection([], **kwargs)
    lc_xyy = mc.LineCollection([], **kwargs)
    ax.add_collection(lc_xxy)
    ax.add_collection(lc_xyy)

    def draw(c):
        lc_xxy.set_segments(forward(xxy, c))
        lc_xyy.set_segments(forward(xyy, c))
        return lc_xxy, lc_xyy

    def init():
        return lc_xxy, lc_xyy

    return init, draw


def plot_background(ax, n=16, m=16, sub=10, φ=IdMap(), c=1):
    colors = np.ones(4) * .2
    delta = 1.15

    init, draw = plot_grid(ax, n=n, m=m, sub=sub, φ=φ, colors=colors, lw=5)
    draw(c)
    ax.set_xlim((-delta, +delta))
    ax.set_ylim((-delta, +delta))


def plot_jacobian(ax, n=16, m=16, sub=10, φ=IdMap(), mode='forward'):
    ax0, ax1 = ax

    plot_background(ax0, n, m, sub, φ, 0)
    plot_background(ax1, n, m, sub, φ, 1)

    s = np.linspace(-1, +1, n)
    t = np.linspace(-1, +1, m)
    s, t = np.meshgrid(s, t)

    f = φ.lambdify()
    x, y = f(s, t)

    kwargs = dict(scale=1.7 * max(n, m))

    o = (0 * s, 0 * s)
    ex = (1 + 0 * s, 0 * s)
    ey = (0 * s, 1 + 0 * s)

    q0x = ax0.quiver(s, t, *o, color='k', **kwargs)
    q0y = ax0.quiver(s, t, *o, color='r', **kwargs)

    q1x = ax1.quiver(x, y, *o, color='k', **kwargs)
    q1y = ax1.quiver(x, y, *o, color='r', **kwargs)

    def set_vec(ux, uy, vx, vy):
        q0x.set_UVC(*ux)
        q0y.set_UVC(*uy)

        q1x.set_UVC(*vx)
        q1y.set_UVC(*vy)

        return q0x, q0y, q1x, q1y

    J = φ.jacobian()
    Jinv = J.inv()

    j = matrixlambdify(φ.xx, J)(s, t)
    jinv = matrixlambdify(φ.xx, Jinv)(s, t)

    R = rotation()

    #  Push forward of basis vectors.
    def draw_forward(θ):
        Rθ = R(θ)
        ux, uy = Rθ(ex), Rθ(ey)
        return set_vec(
            ux, uy,
            j(ux), j(uy)
        )

    # Pull back of basis vectors
    def draw_backwardx(θ):
        Rθ = R(θ)
        ux = normalize(j(ex))
        uy = R(90)(ux)
        vx, vy = Rθ(ux), Rθ(uy)
        return set_vec(
            jinv(vx), jinv(vy),
            vx, vy
        )

    def draw_backwardy(θ):
        Rθ = R(θ)
        uy = normalize(j(ey))
        ux = R(-90)(uy)
        vx, vy = Rθ(ux), Rθ(uy)
        return set_vec(
            jinv(vx), jinv(vy),
            vx, vy
        )

    def init():
        return q0x, q0y, q1x, q1y

    if mode == 'forward':
        draw = draw_forward
    elif mode == 'backwardx':
        draw = draw_backwardx
    elif mode == 'backwardy':
        draw = draw_backwardy
    else:
        raise ValueError('Invalid mode {mode}'.format(mode=mode))

    return init, draw


def plot_jacobian_sqrt(ax, n=16, m=16, sub=10, φ=IdMap()):
    ax0, ax1 = ax

    plot_background(ax0, n, m, sub, φ, 0)
    plot_background(ax1, n, m, sub, φ, 1)

    s = np.linspace(-1, +1, n)
    t = np.linspace(-1, +1, m)
    s, t = np.meshgrid(s, t)

    f = φ.lambdify()
    x, y = f(s, t)

    kwargs = dict(scale=1.7 * max(n, m))

    o = (0 * s, 0 * s)
    ex = (1 + 0 * s, 0 * s)
    ey = (0 * s, 1 + 0 * s)

    q0x = ax0.quiver(s, t, *o, color='k', **kwargs)
    q0y = ax0.quiver(s, t, *o, color='r', **kwargs)

    q1x = ax1.quiver(x, y, *o, color='k', **kwargs)
    q1y = ax1.quiver(x, y, *o, color='r', **kwargs)

    def set_vec(ux, uy, vx, vy):
        q0x.set_UVC(*ux)
        q0y.set_UVC(*uy)

        q1x.set_UVC(*vx)
        q1y.set_UVC(*vy)

        return q0x, q0y, q1x, q1y

    g = φ.metric()
    J = φ.jacobian()

    j = matrixnum(φ.xx, J)(s, t)

    a, b, c, d = sy.symbols('a, b, c, d')
    M = sy.Matrix([[a, b], [c, d]])
    minv = invmat
    msqrt = lambda *args: sqrtmat(*args, module=np)

    mat = matrix()

    gxx, gxy, gyy = g[0, 0], g[0, 1], g[1, 1]
    gxx, gxy, gyy = [sy.lambdify(φ.xx, _, 'numpy')(s, t) for _ in (gxx, gxy, gyy)]
    jginv = minv(*msqrt(gxx, gxy, gxy, gyy))

    ux, uy = mat(jginv)(ex), mat(jginv)(ey)
    vx, vy = mat(j)(ux), mat(j)(uy)

    set_vec(
        ux, uy,
        vx, vy
    )


def plot_hodge_star(ax, n=16, m=16, sub=10, φ=IdMap()):
    ax0, ax1 = ax

    plot_background(ax0, n, m, sub, φ, 0)
    plot_background(ax1, n, m, sub, φ, 1)

    s = np.linspace(-1, +1, n)
    t = np.linspace(-1, +1, m)
    s, t = np.meshgrid(s, t)

    f = φ.lambdify()
    x, y = f(s, t)

    kwargs = dict(scale=1.7 * max(n, m))

    o = (0 * s, 0 * s)
    ex = (1 + 0 * s, 0 * s)

    q0x = ax0.quiver(s, t, *o, color='k', **kwargs)
    q0y = ax0.quiver(s, t, *o, color='r', **kwargs)

    q1x = ax1.quiver(x, y, *o, color='k', **kwargs)
    q1y = ax1.quiver(x, y, *o, color='r', **kwargs)

    def set_vec(ux, uy, vx, vy):
        q0x.set_UVC(*ux)
        q0y.set_UVC(*uy)

        q1x.set_UVC(*vx)
        q1y.set_UVC(*vy)

        return q0x, q0y, q1x, q1y

    JT = φ.jacobian().T
    JTinv = JT.inv()

    JTmat = matrixlambdify(φ.xx, JT)(s, t)
    JTinvmat = matrixlambdify(φ.xx, JTinv)(s, t)

    R = rotation()
    H = R(90)

    def draw(θ):
        Rθ = R(θ)
        ux = Rθ(ex)
        vx = JTinvmat(ux)
        vy = H(vx)
        uy = JTmat(vy)
        return set_vec(ux, uy, vx, vy)

    def init():
        return q0x, q0y, q1x, q1y

    return init, draw


def plot_metric(ax, n=16, m=16, sub=10, φ=IdMap(), mode='flat'):
    ax0, ax1 = ax

    plot_background(ax0, n, m, sub, φ, 0)
    plot_background(ax1, n, m, sub, φ, 0)

    s = np.linspace(-1, +1, n)
    t = np.linspace(-1, +1, m)
    s, t = np.meshgrid(s, t)

    g = φ.metric()
    if mode == 'flat':
        M = g
    elif mode == 'sharp':
        M = g.inv()
    else:
        raise ValueError

    Mmat = matrixlambdify(φ.xx, M)(s, t)

    kwargs = dict(scale=1.7 * max(n, m))

    o = (0 * s, 0 * s)
    ex = (1 + 0 * s, 0 * s)
    ey = (0 * s, 1 + 0 * s)

    q0x = ax0.quiver(s, t, *o, color='k', **kwargs)
    q0y = ax0.quiver(s, t, *o, color='r', **kwargs)

    q1x = ax1.quiver(s, t, *o, color='k', **kwargs)
    q1y = ax1.quiver(s, t, *o, color='r', **kwargs)

    def set_vec(ux, uy, vx, vy):
        q0x.set_UVC(*ux)
        q0y.set_UVC(*uy)

        q1x.set_UVC(*vx)
        q1y.set_UVC(*vy)

        return q0x, q0y, q1x, q1y

    R = rotation()

    def draw(θ):
        Rθ = R(θ)
        ux = Rθ(ex)
        uy = Rθ(ey)

        vx = Mmat(Rθ(ex))
        vy = Mmat(Rθ(ey))

        return set_vec(ux, uy, vx, vy)

    def init():
        return q0x, q0y, q1x, q1y

    return init, draw


if __name__ == '__main__':
    from spexy.morphism.maps import get_map

    φ = get_map('gerritsma')

    # fig = plt.figure()
    # ax = plt.gca()
    # init, draw = plot_grid(ax, φ=φ)
    # draw(0.5)

    h = 10
    fig, ax = plt.subplots(1, 2, figsize=(2 * h, h))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    init, draw = plot_jacobian(ax, φ=φ, mode='backwardx')
    draw(0)

    plt.show()
