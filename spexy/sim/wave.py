from collections import namedtuple
from importlib.machinery import SourceFileLoader

import numpy as np
import sympy as sy

from spexy.diffgeom import PullBack, Tang, Inv
from spexy.form import sym
from spexy.helper import peek, logistic
from spexy.opengl.scalarfield import scalarfield
from spexy.opengl.surface import surface
from spexy.sim.save import save, saverun
from glumpy import app

Const = namedtuple('Const', ['x', 'y', 'deltax', 'deltay'])
Var = namedtuple('Var', ['z', 'Jzu', 'Jzv'])


@save(Const, Var)
def diffusion_sim(
        g, M, s,
        h=1e-4,
        nsteps=10,
):
    x, y = sy.symbols('u, v')
    pull = PullBack(M)

    F0 = sym.Form.forms(x, y)[0]
    f = F0(pull(s)(x, y))
    f = f.P(g)

    ((X, Y),) = g.verts()
    (X, Y) = sy.lambdify([x, y], M(x, y), 'numpy')(X, Y)

    deltax = np.diff(X[:, 0])
    deltay = np.diff(Y[0, :])

    yield Const(x=X, y=Y, deltax=deltax, deltay=deltay)

    vx, vy = sy.symbols('vx, vy')
    xx, yy, tx, ty = Tang(Inv(M))(x, y, vx, vy)
    tang = sy.lambdify([x, y, vx, vy], [tx, ty], 'numpy')

    while True:
        fx, fy = f.G.comp
        z = f.array[0]
        Jzu, Jzv = tang(X, Y, fx.array[0], fy.array[0])
        yield Var(z=z, Jzu=Jzu, Jzv=Jzv)

        for _ in range(nsteps):
            ft = f.D.H.D.H
            f += h * ft


@save(Const, Var)
def wavelinear_sim(
        g, M, s,
        h=1e-4,
        nsteps=10,
        ic='x',
):
    x, y = sy.symbols('u, v')
    pull = PullBack(M)

    F0 = sym.Form.forms(x, y)[0]
    sref = pull(s)
    f = F0(sref(x, y))
    ic_opt = {
        'x': lambda: F0(-sy.diff(sref(x, y), x)),
        '0': lambda: F0(0),
        '+r': lambda: F0(+ sy.sqrt(sy.diff(sref(x, y), x) ** 2 + sy.diff(sref(x, y), y) ** 2)),
        '-r': lambda: F0(- sy.sqrt(sy.diff(sref(x, y), x) ** 2 + sy.diff(sref(x, y), y) ** 2)),
    }
    ft = ic_opt[ic]()

    f = f.P(g)
    ft = ft.P(g)

    ((X, Y),) = g.verts()
    (X, Y) = sy.lambdify([x, y], M(x, y), 'numpy')(X, Y)

    deltax = np.diff(X[:, 0])
    deltay = np.diff(Y[0, :])

    yield Const(x=X, y=Y, deltax=deltax, deltay=deltay)

    vx, vy = sy.symbols('vx, vy')
    xx, yy, tx, ty = Tang(Inv(M))(x, y, vx, vy)
    tang = sy.lambdify([x, y, vx, vy], [tx, ty], 'numpy')

    while True:
        fx, fy = f.G.comp
        z = f.array[0]
        Jzu, Jzv = tang(X, Y, fx.array[0], fy.array[0])
        yield Var(z=z, Jzu=Jzu, Jzv=Jzv)

        for _ in range(nsteps):
            ftt = f.D.H.D.H
            ft += h * ftt
            f += h * ft


@saverun
def wavelinear_glpy(sim, view):
    def SJ(x, y, z, Jzu, Jzv):
        S = [x, y, 0.5*z]
        J = [1, 0, 0, 1, 0.5*Jzu, 0.5*Jzv]
        J = np.broadcast_arrays(*J)
        return S, J

    const = next(sim)
    sim, step = peek(sim)

    surf = surface(
        *SJ(const.x, const.y, step.z, step.Jzu, step.Jzv),
        color=(1, 1, 1, 1),
        deltax=const.deltax, deltay=const.deltay,
    )

    w = app.Window(width=800, height=800)
    w.attach(surf.transform)
    view(surf.transform)

    @w.event
    def on_init():
        surf.on_init()

    @w.event
    def on_draw(_):
        step = next(sim, None)
        if step is None:
            app.quit()
            return
        surf.on_draw(*SJ(const.x, const.y, step.z, step.Jzu, step.Jzv))

    return w


@saverun
def wavelinear_cm(sim):
    const = next(sim)
    sim, step = peek(sim)

    k = 5
    s = logistic(k, 0)

    field = scalarfield(s(step.z), deltax=const.deltax, deltay=const.deltay, cmap='icefire')

    w = app.Window(width=800, height=800)

    @w.event
    def on_draw(_):
        step = next(sim, None)
        if step is None:
            app.quit()
            return
        field.on_draw(s(step.z))

    return w


frames = 1000


def sim_save(source, dest):
    sim = SourceFileLoader('_', source).load_module().sim
    sim.write(dest, frames)


def sim_glpy(source, module, dest):
    module = SourceFileLoader('_', module).load_module()
    sim = wavelinear_sim(source)
    if hasattr(module, 'view'):
        view = module.view
    else:
        view = lambda _: None
    run, save = wavelinear_glpy(sim, view)
    save(dest)


def sim_cm(source, module, dest):
    sim = wavelinear_sim(source)
    run, save = wavelinear_cm(sim)
    save(dest)


if __name__ == '__main__':
    import argh

    parser = argh.ArghParser()
    parser.add_commands([
        sim_save, sim_glpy, sim_cm
    ])
    parser.dispatch()


def ic_gauss():
    c = sy.Rational(1, 6)
    s = lambda x, y: sy.exp(-(x ** 2 + y ** 2) / 2 / c ** 2)
    return s


def ic_logistic():
    def f(x, k, x0):
        return 1 / (1 + sy.exp(-k * (x - x0)))

    def g(x, k):
        return (
            f(+x, k, -sy.Rational(1, 4)) *
            f(-x, k, -sy.Rational(1, 4))
        )

    k = 32
    l = lambda x: g(x, k)

    def s(x, y):
        # x += sy.Rational(1, 3)
        # y -= sy.Rational(1, 3)
        return l(x) * l(y)

    # s = lambda x, y: l(sy.sqrt(x ** 2 + y ** 2))
    return s


def ic_logistic_corner():
    def f(x, k, x0):
        return 1 / (1 + sy.exp(-k * (x - x0)))

    def g(x, k=32, r=sy.Rational(4, 5)):
        return (
            f(+x, k, -r) *
            f(-x, k, -r)
        )

    l = lambda x: g(x, r=sy.Rational(2, 4))

    def s(x, y):
        x += sy.Rational(1, 3)
        y -= sy.Rational(1, 3)
        return l(sy.sqrt(x ** 2 + y ** 2)) / 3

    return s
