from collections import namedtuple
from importlib.machinery import SourceFileLoader

import numpy as np
import sympy as sy

from spexy.embed.embed import sqrt_g
from spexy.embed.jacob import SJ
from spexy.form import sym, coch
from spexy.helper import logistic, peek
from spexy.morphism.matrix import invmat
from spexy.sim.save import save, saverun


def bases_sqrt(*J):
    J = sqrt_g(*J)
    Jinv = invmat(*J)
    return J, Jinv


def form_center(g):
    x, y = sy.symbols('x, y')

    c = sy.Rational(1, 8)
    scalar = lambda x, y: sy.exp(-(x ** 2 + y ** 2) / 2 / c ** 2) * 2

    F0 = sym.Form.forms(x, y)[0]
    f = F0(scalar(x, y))
    f = f.P(g)
    return f


def form(g):
    x, y = sy.symbols('x, y')

    c = sy.Rational(1, 16)
    scalar = lambda x, y: sy.exp(-((x - 1) ** 2) / 2 / c ** 2) * 2

    F0 = sym.Form.forms(x, y)[0]
    f = F0(scalar(x, y))
    f = f.P(g)
    return f


Const = namedtuple('Const', ['deltax', 'deltay', 'St', 'Jt'])
Var = namedtuple('Var', ['time', 'f'])


@save(Const, Var)
def surface_form_sim(s, g, bases=bases_sqrt, h=1e-4, nsteps=20, flat=False, form=form):
    time = 0.0

    sj = SJ(s, g)
    St, Jt = sj(time)

    ((x, y),) = g.verts()
    deltax, deltay = np.diff(x[:, 0]), np.diff(y[0, :])
    yield Const(deltax=deltax, deltay=deltay, St=St, Jt=Jt)

    J, Jinv = bases(*Jt)
    J = [coch.Form(0, g, [_]).Pup.comp[0].array for _ in J]
    Jinv = [coch.Form(0, g, [_]).Pup.comp[0].array for _ in Jinv]

    f = form(g)
    ft = 0 * f

    while True:

        yield Var(time=time, f=f.array[0])

        for _ in range(nsteps):
            if flat:
                ftt = f.D.H.D.H
            else:
                ftt = f.D.HH(J, Jinv).D.HH(J, Jinv)
            ft += h * ftt
            f += h * ft
            time += h


def view(_):
    _.phi = 180
    _.theta = 45
    _.zoom = 30
    _.distance = 8


@saverun
def surface_form_glpy(sim, view=view):
    from spexy.opengl.scalarfield import scalarfield
    from spexy.opengl.surface import surface
    from glumpy import app, gloo, gl
    w = app.Window(800, 800, color=(1, 1, 1, 1))

    cwidth, cheight = 2 * w.width, 2 * w.height
    texture = np.zeros((cwidth, cheight, 4), dtype=np.float32).view(gloo.TextureFloat2D)
    framebuffer = gloo.FrameBuffer(color=[texture])

    const = next(sim)
    sim, step = peek(sim)

    rf = logistic(3, 0)
    field = scalarfield(
        rf(step.f),
        deltax=const.deltax, deltay=const.deltay,
    )

    surf = surface(
        const.St, const.Jt,
        texture=texture, mesh=False,
        deltax=const.deltax, deltay=const.deltay,
    )

    w.attach(surf.transform)
    view(surf.transform)

    @w.event
    def on_init():
        surf.on_init()
        on_draw(0)

    @w.event
    def on_draw(_):
        step = next(sim, None)
        if step is None:
            app.quit()
            return

        framebuffer.activate()

        gl.glViewport(0, 0, cwidth, cheight)
        field.on_draw(rf(step.f))

        framebuffer.deactivate()

        gl.glViewport(0, 0, w.width, w.height)
        surf.on_draw(const.St, const.Jt)

    return w


frames = 1000


def sim_save(source, dest):
    sim = SourceFileLoader('_', source).load_module().sim
    sim.write(dest, frames)


def sim_glpy(source, module, dest):
    module = SourceFileLoader('_', module).load_module()
    sim = surface_form_sim(source)
    if hasattr(module, 'view'):
        view_ = module.view
    else:
        view_ = view
    run, save = surface_form_glpy(sim, view_)
    save(dest)


if __name__ == '__main__':
    import argh

    parser = argh.ArghParser()
    parser.add_commands([
        sim_save, sim_glpy
    ])
    parser.dispatch()
