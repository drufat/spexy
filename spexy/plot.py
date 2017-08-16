# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from licpy.pixelize import pixelize_endpoints, pixelize

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'


def plot_bases_1d(plt, g, xmin, xmax, name):
    N = len(g.verts())
    z = np.linspace(xmin, xmax, 100)
    B0, B1, B0d, B1d = g.basis_fn()
    N0, N1, N0d, N1d = g.numbers()
    x, = g.verts()
    xd, = g.dual.verts()

    plt.subplot(221)
    plt.title(r"${%s}^0_{%d,n}$" % (name, N))
    for i in range(N0):
        plt.plot(z, B0(i)(z), label=str(i))
        plt.scatter(x, 0 * x)

    plt.subplot(222)
    plt.title(r"$\widetilde{%s}^0_{%d,n}$" % (name, N))
    for i in range(N0d):
        plt.plot(z, B0d(i)(z), label=str(i))
        plt.scatter(x, 0 * x)
        plt.scatter(xd, 0 * xd, color='red', marker='x')

    plt.subplot(223)
    plt.title(r"$\widetilde{{%s}}^1_{%d,n}$" % (name, N))
    for i in range(N1d):
        plt.plot(z, B1d(i)(z) * g.dual.delta[1](i), label=str(i))
        plt.scatter(x, 0 * x)
        plt.scatter(xd, 0 * xd, color='red', marker='x')

    plt.subplot(224)
    plt.title(r"${%s}^1_{%d,n}$" % (name, N))
    for i in range(N1):
        plt.plot(z, B1(i)(z) * g.delta[1](i), label=str(i))
        plt.scatter(x, 0 * x)


def grid_1d(ax, g):
    vertices_1d(ax, g)
    edges_1d(ax, g)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')


def vertices_1d(ax, g):
    x, = g.verts()
    ax.scatter(x, 0 * x, color='k')


def edges_1d(ax, g):
    for x0, x1 in zip(*g.edges()):
        ax.plot((x0, x1), (0, 0),
                color='r', linewidth=3, zorder=0)


def avg(x):
    return sum(x) / len(x)


class PlotGrid1D():
    def __init__(self, g, labels=True, drawborder=True, ymin=-.1, ymax=.7):
        def axis(ax):
            ax.axis('off')
            Δ = .02 * np.abs(g.xmax - g.xmin)
            ax.axis([g.xmin - Δ, g.xmax + Δ, ymin, ymax])

        def border(ax):
            if not drawborder: return
            ax.plot([g.xmin, g.xmin], [ymin, ymax], '--', color='k', alpha=0.8, lw=1)
            ax.plot([g.xmax, g.xmax], [ymin, ymax], '--', color='k', alpha=0.8, lw=1)

        def annotate(ax, prefix, i, xy, textoffset, color):
            if not labels: return
            xytext = (0, textoffset)
            ax.annotate(
                '${prefix}_{{ {i} }}$'.format(prefix=prefix, i=i),
                xy=xy, xytext=xytext, size=20, color=color,
                textcoords='offset points',
                horizontalalignment='center', verticalalignment='center',
            )

        def verts(ax, g, prefix, textoffset=20, color='k', x=0, y=0):
            axis(ax)

            X = g.verts()
            Xc = avg(X)

            for i, xc in enumerate(Xc):
                x0 = X[0][i]
                ax.plot([x + x0], [y], marker='.', c=color, ms=20)
                annotate(ax, prefix, i, (x + xc, y), textoffset, color)

            if g.bndry:
                ax.plot([x + g.xmin], [y], marker='o', c='white', ms=8, mec=color, mew=2)
                ax.plot([x + g.xmax], [y], marker='o', c='white', ms=8, mec=color, mew=2)

        def edges(ax, g, prefix, textoffset=20, color='k', x=0, y=0):
            axis(ax)

            X = g.edges()
            Xc = avg(X)

            for i, xc in enumerate(Xc):
                gap = 0.089
                x0, x1 = X[0][i], X[1][i]
                x0 += gap
                x1 -= gap
                ax.plot([x + x0, x + x1], [y, y], '-', color=color, alpha=0.8, lw=4)
                annotate(ax, prefix, i, (x + xc, y), textoffset, color)

        self.border = lambda ax: border(ax)
        self.V = lambda ax, x=0, y=.5: verts(ax, g, r'\bar{V}', textoffset=20, color='k', x=x, y=y)
        self.E = lambda ax, x=0, y=.5: edges(ax, g, r'\bar{E}', textoffset=-20, color='k', x=x, y=y)
        self.Vd = lambda ax, x=0, y=0: verts(ax, g.dual, r'\widetilde{V}', textoffset=-22, color='r', x=x, y=y)
        self.Ed = lambda ax, x=0, y=0: edges(ax, g.dual, r'\widetilde{E}', textoffset=18, color='r', x=x, y=y)

    def plot(self, ax):
        self.border(ax)
        self.E(ax)
        self.Ed(ax)
        self.V(ax)
        self.Vd(ax)


class PlotGrid(object):
    def __init__(self, g, labels=True, bndry=False, egap=0.08, fgap=0.03, labelsize=20):
        def axis(ax):
            ε = .1
            ax.axis('equal')
            ax.axis('off')
            ax.axis((
                g.blk.gx.xmin - ε,
                g.blk.gx.xmax + ε,
                g.blk.gy.xmin - ε,
                g.blk.gy.xmax + ε,
            ))

        def plot_bndry(ax):
            if not bndry: return
            xmin = g.blk.gx.xmin
            xmax = g.blk.gx.xmax
            ymin = g.blk.gy.xmin
            ymax = g.blk.gy.xmax
            ax.plot((xmin, xmax, xmax, xmin, xmin),
                    (ymin, ymin, ymax, ymax, ymin), '-', c='k', lw=0.3, alpha=0.3)

        def annotate(ax, xy, prefix, i,
                     xytext=(0, 0),
                     bbox=dict(boxstyle='round4,pad=0.2', fc='w', ec='w', lw=0, alpha=0.6)):
            if not labels: return
            ax.annotate(
                '${prefix}_{{ {i} }}$'.format(prefix=prefix, i=i),
                xy=xy, xytext=xytext, size=labelsize,
                textcoords='offset points',
                horizontalalignment='center', verticalalignment='center',
                bbox=bbox
            )

        def verts(ax, g, prefix='', textoffset=labelsize, color='k'):
            pnts = g.verts()
            X, Y = pnts[0::2], pnts[1::2]
            Xc, Yc = avg(X), avg(Y)
            for i, (xc, yc) in enumerate(zip(Xc, Yc)):
                x0 = X[0][i]
                y0 = Y[0][i]
                ax.scatter([x0], [y0], color=color, s=64)
                annotate(ax, (xc, yc), prefix, i,
                         xytext=(0, textoffset),
                         bbox={})
            axis(ax)
            plot_bndry(ax)

        self.V = lambda ax: verts(ax, g, prefix=r'\bar{V}', color='k')
        self.Vd = lambda ax: verts(ax, g.dual, prefix=r'\widetilde{V}', color='r')

        def verts_bndry(ax, g, color='k'):
            if g.blk.gx.bndry:
                for y in [g.blk.gy.xmin, g.blk.gy.xmax]:
                    xv, = g.blk.gx.verts()
                    yv = np.zeros_like(xv) + y
                    ax.scatter(xv, yv, color='none', edgecolors=color, s=48, marker='o', linewidth=1)
            if g.blk.gy.bndry:
                for x in [g.blk.gx.xmin, g.blk.gx.xmax]:
                    yv, = g.blk.gy.verts()
                    xv = np.zeros_like(yv) + x
                    ax.scatter(xv, yv, color='none', edgecolors=color, s=48, marker='o', linewidth=1)
            axis(ax)
            plot_bndry(ax)

        self.V_bndry = lambda ax: verts_bndry(ax, g, color='k')
        self.Vd_bndry = lambda ax: verts_bndry(ax, g.dual, color='r')

        @np.vectorize
        def fixgap(x0, y0, x1, y1):
            gap = egap
            if np.isclose(x0, x1):
                y0 += gap
                y1 -= gap
            if np.isclose(y0, y1):
                x0 += gap
                x1 -= gap
            return x0, y0, x1, y1

        def edges(ax, g, prefix='', color='k'):
            pnts = g.edges()
            X, Y = pnts[0::2], pnts[1::2]
            Xc, Yc = avg(X), avg(Y)

            for i, (xc, yc) in enumerate(zip(Xc, Yc)):
                x0, x1 = X[0][i], X[1][i]
                y0, y1 = Y[0][i], Y[1][i]
                x0, y0, x1, y1 = fixgap(x0, y0, x1, y1)
                ax.plot([x0, x1], [y0, y1], '-', color=color, alpha=0.8, lw=4)
                annotate(ax, (xc, yc), prefix, i)
            axis(ax)
            plot_bndry(ax)

        self.E = lambda ax: edges(ax, g, prefix=r'\bar{E}', color='k')
        self.Ed = lambda ax: edges(ax, g.dual, prefix=r'\widetilde{E}', color='r')

        def edges_bndry(ax, g, color='k'):

            if g.blk.gx.bndry:
                for y in [g.blk.gy.xmin, g.blk.gy.xmax]:
                    x0, x1 = g.blk.gx.edges()
                    y0 = np.zeros_like(x0) + y
                    y1 = np.zeros_like(x0) + y
                    x0, y0, x1, y1 = fixgap(x0, y0, x1, y1)
                    ax.plot([x0, x1], [y0, y1], '--', color=color, alpha=0.8, lw=4)
            if g.blk.gy.bndry:
                for x in [g.blk.gx.xmin, g.blk.gx.xmax]:
                    y0, y1 = g.blk.gy.edges()
                    x0 = np.zeros_like(y0) + x
                    x1 = np.zeros_like(y0) + x
                    x0, y0, x1, y1 = fixgap(x0, y0, x1, y1)
                    ax.plot([x0, x1], [y0, y1], '--', color=color, alpha=0.8, lw=4)
            axis(ax)
            plot_bndry(ax)

        self.E_bndry = lambda ax: edges_bndry(ax, g, color='k')
        self.Ed_bndry = lambda ax: edges_bndry(ax, g.dual, color='r')

        def faces(ax, g, prefix='', color='k'):
            pnts = g.faces()
            X, Y = pnts[0::2], pnts[1::2]
            Xc, Yc = avg(X), avg(Y)

            def fixgap(x0, y0, x1, y1, x2, y2, x3, y3):
                gap = fgap
                x0 += gap
                y0 += gap
                x1 -= gap
                y1 += gap
                x2 -= gap
                y2 -= gap
                x3 += gap
                y3 -= gap
                return x0, y0, x1, y1, x2, y2, x3, y3

            for i, (xc, yc) in enumerate(zip(Xc, Yc)):
                x0, x1, x2, x3 = X[0][i], X[1][i], X[2][i], X[3][i]
                y0, y1, y2, y3 = Y[0][i], Y[1][i], Y[2][i], Y[3][i]
                x0, y0, x1, y1, x2, y2, x3, y3 = fixgap(x0, y0, x1, y1, x2, y2, x3, y3)
                p = mpl.patches.Polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], color=color, alpha=0.5)
                ax.add_patch(p)
                annotate(ax, (xc, yc), prefix, i)
            axis(ax)
            plot_bndry(ax)

        self.F = lambda ax: faces(ax, g, prefix=r'\bar{F}', color='k')
        self.Fd = lambda ax: faces(ax, g.dual, prefix=r'\widetilde{F}', color='r')

        def points(ax, verts):
            x, y = verts
            ax.plot(x, y, '.', ms=12, c='k')
            axis(ax)
            plot_bndry(ax)

        gg = g.refine()
        self.P = lambda ax: points(ax, gg.verts())


class PlotComponent(object):
    def __init__(self, g):
        def axis(ax):
            ε = .5
            ax.axis('equal')
            ax.axis('off')
            ax.axis((
                g.blk.gx.xmin - ε,
                g.blk.gx.xmax + ε,
                g.blk.gy.xmin - ε,
                g.blk.gy.xmax + ε,
            ))

        def verts(ax, g, color):
            pnts = g.verts()
            X, Y = pnts[0::2], pnts[1::2]
            Xc, Yc = avg(X), avg(Y)
            for i, (xc, yc) in enumerate(zip(Xc, Yc)):
                def vel():
                    xmin, xmax = g.blk.gx.xmin, g.blk.gx.xmax
                    ymin, ymax = g.blk.gy.xmin, g.blk.gy.xmax
                    x = 2 * (xc - xmin) / (xmax - xmin) - 1
                    y = 2 * (yc - ymin) / (ymax - ymin) - 1
                    vx = -y / np.sqrt(x ** 2 + y ** 2)
                    vy = +x / np.sqrt(x ** 2 + y ** 2)
                    if np.isnan(vx): vx = 0.0
                    if np.isnan(vy): vy = 0.0
                    vx *= 0.3
                    vy *= 0.3
                    vx += 0.05
                    vy += 0.07
                    return vx, vy

                vx, vy = vel()
                hl = 0.08
                hw = 0.05
                ax.arrow(xc, yc, vx, vy, fc=color, ec=color, head_width=hw, head_length=hl, linewidth=1, zorder=1)
                ax.arrow(xc, yc, vx, 00, fc='grey', ec='grey', head_width=hw, head_length=hl, linewidth=0.3, zorder=1)
                ax.arrow(xc, yc, 00, vy, fc='grey', ec='grey', head_width=hw, head_length=hl, linewidth=0.3, zorder=1)
                ax.scatter([xc], [yc], facecolors='white', edgecolors=color, s=32, marker='o', linewidth=1, zorder=2)
            axis(ax)

        self.V = lambda ax: verts(ax, g, color='k')
        self.Vd = lambda ax: verts(ax, g.dual, color='r')


def interp2D(N, M):
    def _(f):
        ff = f.blk.Pup
        a = [_.array[0] for _ in ff.comp]
        [[x, y], ] = ff.comp[0].grid.verts()
        rslt = tuple([x, y]) + tuple(a)

        if f.degree == 0:
            pix = pixelize_endpoints
        if f.degree == 1:
            pix = pixelize

        return tuple(
            pix(N + 1, M + 1, x[:, 0], y[0, :], _)
            for _ in rslt
        )

    return _


_interp0 = interp2D(128, 128)
_interp1 = interp2D(8, 8)


def plotf(f, **kwargs):
    if f.degree == 0:
        plt.contourf(*_interp0(f), cmap=plt.cm.terrain, **kwargs)
    if f.degree == 1:
        plt.quiver(*_interp1(f), **kwargs)


if __name__ == '__main__':
    from spexy.grid import Grid_1D, Grid_2D

    plt.figure()
    ax = plt.subplot(311)
    g = Grid_1D.chebyshev(3)
    grid_1d(ax, g)

    ax = plt.subplot(312)
    g = Grid_1D.chebyshev(5)
    grid_1d(ax, g)

    ax = plt.subplot(313)
    g = Grid_1D.chebyshev(15)
    grid_1d(ax, g)

    plt.figure()
    g = Grid_1D.chebyshev(5)
    plot_bases_1d(plt, g, -1, 1, "\psi")

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    g = PlotGrid(Grid_2D.regnew(3, 3), labels=False, bndry=True)
    g.V(ax[0, 0])
    g.E(ax[0, 1])
    g.F(ax[0, 2])
    g.Fd(ax[1, 0])
    g.Ed(ax[1, 1])
    g.Vd(ax[1, 2])
    fig.tight_layout()

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    g = PlotGrid(Grid_2D.regnew(3, 3), bndry=True)
    g.V(ax[0, 0])
    g.E(ax[0, 1])
    g.F(ax[0, 2])
    g.Fd(ax[1, 0])
    g.Ed(ax[1, 1])
    g.Vd(ax[1, 2])
    fig.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    g = PlotComponent(Grid_2D.regular(3, 3))
    g.V(ax)
    fig.tight_layout()

    plt.show()
