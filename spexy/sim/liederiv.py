# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from collections import namedtuple
from importlib.machinery import SourceFileLoader

import numpy as np
import sympy as sy

from spexy.form.sym import Form
from spexy.helper import peek
from spexy.sim.save import save, saverun


class viewport:
    def __init__(self, g):
        if hasattr(g, 'blk'):
            g = g.blk
        xmin, xmax = g.gx.xmin, g.gx.xmax
        ymin, ymax = g.gy.xmin, g.gy.xmax

        def P(x, y):
            x = (x - xmin) / (xmax - xmin)
            y = (y - ymin) / (ymax - ymin)
            x = 2 * x - 1
            y = 2 * y - 1
            return x, y

        def Pinv(x, y):
            x = (x + 1) / 2
            y = (y + 1) / 2
            x = xmin * (1 - x) + xmax * x
            y = ymin * (1 - y) + ymax * y
            return x, y

        sx = (xmax - xmin) / 2
        sy = (ymax - ymin) / 2

        def T(vx, vy):
            return sx * vx, sy * vy

        def Tinv(vx, vy):
            return vx / sx, vy / sy

        self.P = P
        self.Pinv = Pinv
        self.T = T
        self.Tinv = Tinv


x, y = sy.symbols('x, y')
F1 = Form.forms(x, y)[1]


def courant(u, h, g):
    ux, uy = [_.array[0] for _ in u.Pup.comp]
    ux = np.max(np.abs(ux))
    uy = np.max(np.abs(uy))
    dx = g.gx.delta[1](0)
    dy = g.gy.delta[1](0)
    dt = h
    return (ux / dx + uy / dy) * dt


Const = namedtuple('Const', ['xy', 'cr', 'u'])
Var = namedtuple('Var', ['t', 'v'])


@save(Const, Var)
def lieadvect_sim(g, U, V, h=1e-3, nsteps=10):
    xy = g.refine().verts()[0]
    view = viewport(g)
    u = F1(*view.T(*U(*view.P(x, y)))).P(g)
    v = F1(*view.T(*V(*view.P(x, y)))).P(g)

    vec_u = view.Tinv(*[_.array[0] for _ in u.Pup.comp])
    cr = courant(u, h, g)

    yield Const(xy=xy, u=vec_u, cr=cr)

    t = 0.0
    while True:
        vec_v = view.Tinv(*[_.array[0] for _ in v.Pup.comp])

        yield Var(t=t, v=vec_v)

        for _ in range(nsteps):
            dv = -u.Lie(v)
            v += h * dv
            t += h


@save(Const, Var)
def lieadvect_self_sim(g, V, h=1e-3, nsteps=10):
    xy = g.refine().verts()[0]
    view = viewport(g)
    v = F1(*view.T(*V(*view.P(x, y)))).P(g)

    vec_u = np.array([np.zeros_like(_.array[0]) for _ in v.Pup.comp])
    cr = courant(v, h, g)

    yield Const(xy=xy, u=vec_u, cr=cr)

    t = 0.0
    while True:
        vec_v = view.Tinv(*[_.array[0] for _ in v.Pup.comp])

        yield Var(t=t, v=vec_v)

        for _ in range(nsteps):
            dv = -v.Lie(v)
            v += h * dv
            t += h


def liederiv_mplt(sim):
    const = next(sim)
    sim, step = peek(sim)

    def txt(t):
        return "Cr={:.3f} t={:1.3f}".format(float(const.cr), float(t))

    from matplotlib import pyplot as plt, animation
    from spexy.matplot import MatplotAnimation
    fig = plt.figure(figsize=(4, 4), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    a = {}

    def init():
        a['u'] = ax.quiver(*const.xy, *const.u, color='r', scale=30)
        a['v'] = ax.quiver(*const.xy, *step.v, color='k', scale=30)
        a['t'] = ax.text(.1, .1, txt(step.t), fontsize=15, backgroundcolor='w')
        return a['u'], a['v'], a['t']

    def anim(_):
        step = next(sim)
        a['v'].set_UVC(*step.v)
        a['t'].set_text(txt(step.t))
        return a['u'], a['v'], a['t']

    def run():
        _ = animation.FuncAnimation(
            fig, anim,
            init_func=init,
            blit=False,
        )
        plt.show()

    def save(dest):
        fps = 30
        MatplotAnimation(
            animation.FuncAnimation(
                fig, anim,
                init_func=init,
                #frames=frames,
                interval=1000.0 / fps,
                blit=False,
            )
        ).save(dest)

    return run, save


@saverun
def liedriv_glumpy(sim):
    const = next(sim)
    sim, step = peek(sim)

    from glumpy import app
    from spexy.opengl.vectorfields import vectorfields

    vec = vectorfields(
        [const.u, step.v],
        rows=18, cols=18,
        linewidth=3.5,
        colors=[
            (1, 0, 0, 1),
            (0, 0, 0, 1),
        ],
    )

    window = app.Window(width=512, height=512, color=(1, 1, 1, 1))

    @window.event
    def on_draw(_):
        step = next(sim, None)
        if step is None:
            app.quit()
            return
        vec.on_draw([const.u, step.v])

    @window.event
    def on_resize(width, height):
        vec.on_resize(width, height)

    return window


frames = 200


def sim_save(source, dest):
    sim = SourceFileLoader('_', source).load_module().sim
    sim.write(dest, frames)


def sim_mplt(source, dest):
    sim = lieadvect_sim(source)
    run, save = liederiv_mplt(sim)
    save(dest)


def sim_glpy(source, dest):
    sim = lieadvect_sim(source)
    run, save = liedriv_glumpy(sim)
    save(dest)


def lic(source, dest):
    _, vname, i, _ = dest.split('.')
    import h5py
    with h5py.File(source, 'r') as f:
        if vname == 'u':
            vx, vy = f[vname][...]
        else:
            vx, vy = f[vname][int(i)]
        x, y = f['xy'][...]

    N, M = 128, 128
    L = N // 10

    from licpy.lic import runlic_resample
    tex_out = runlic_resample(N, M, x[:, 0], y[0, :], vx, vy, L)
    from spexy.matplot import grey_save
    grey_save(dest, tex_out)


if __name__ == '__main__':
    import argh

    parser = argh.ArghParser()
    parser.add_commands([
        lic, sim_save, sim_mplt, sim_glpy
    ])
    parser.dispatch()
