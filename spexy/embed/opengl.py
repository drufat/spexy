# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from collections import namedtuple
from functools import wraps
from importlib import import_module

import numpy as np

from spexy.command.anim import glumpyanim
from spexy.embed.embed import bases_id, bases_sqrt
from spexy.embed.jacob import SJ, SJ_sym
from spexy.grid import GridBlock_2D
from spexy.opengl.surface import surface
from spexy.opengl.vectorfields import vectorfields
from glumpy import gloo, gl, app


def surface_glpy(w, s, N, theta):
    u = -np.cos(np.linspace(0, np.pi, N + 1))
    v = -np.cos(np.linspace(0, np.pi, N + 1))
    U, V = np.meshgrid(u, v, indexing='ij')

    sj = SJ_sym(s, U, V)

    t = 0.0
    dt = 1.5e-2
    surf = surface(
        *sj(t),
        color=(1, 1, 1, 1),
    )

    w.attach(surf.transform)
    surf.transform.phi = 0.0
    surf.transform.theta = theta

    @w.event
    def on_init():
        surf.on_init()
        surf.on_draw(*sj(t))

    @w.event
    def on_draw(_):
        nonlocal t
        surf.on_draw(*sj(t))
        t += dt


def surface_vector(w, s, g, bases):
    cwidth, cheight = 2 * w.width, 2 * w.height
    texture = np.zeros((cwidth, cheight, 4), dtype=np.float32).view(gloo.TextureFloat2D)
    framebuffer = gloo.FrameBuffer(color=[texture])

    t = 0.0
    dt = 1.5e-2

    sj = SJ(s, g)
    St, Jt = sj(t)
    e0, e1 = bases(*Jt)

    vec = vectorfields(
        [e0, e1],
        colors=[(0, 0, 0, 1), (1, 0, 0, 1)],
        deltax=g.gx.deltas[1:-1], deltay=g.gy.deltas[1:-1],
        rows=13, cols=13,
        linewidth=5,
    )

    surf = surface(
        St, Jt,
        texture=texture, mesh=False,
        deltax=g.gx.deltas[1:-1], deltay=g.gy.deltas[1:-1],
    )

    w.attach(surf.transform)
    _ = surf.transform
    _.phi = 0

    @w.event
    def on_init():
        surf.on_init()
        on_draw(0)

    @w.event
    def on_draw(_):
        nonlocal t

        St, Jt = sj(t)
        e0, e1 = bases(*Jt)

        framebuffer.activate()

        gl.glViewport(0, 0, cwidth, cheight)
        vec.on_draw([e0, e1])
        vec.on_resize(cwidth, cheight)

        framebuffer.deactivate()

        gl.glViewport(0, 0, w.width, w.height)
        surf.on_draw(St, Jt)

        t += dt


def flat_vector(w, s, g, bases):
    sj = SJ(s, g)

    time = 0.0
    dt = 1.5e-2

    St, Jt = sj(time)
    e0, e1 = bases_sqrt(*Jt)

    vec = vectorfields(
        [e0, e1],
        colors=[(0, 0, 0, 1), (1, 0, 0, 1)],
        deltax=g.gx.deltas[1:-1], deltay=g.gy.deltas[1:-1],
        rows=13, cols=13,
        linewidth=3,
    )

    @w.event
    def on_init():
        on_draw(0)

    @w.event
    def on_draw(_):
        nonlocal time

        St, Jt = sj(time)
        e0, e1 = bases(*Jt)

        vec.on_draw([e0, e1])

        time += dt

    @w.event
    def on_resize(width, height):
        vec.on_resize(width, height)


def genembed(name, dest, gen):
    s = import_module("spexy.embed.surface.{}".format(name)).s

    N = 48
    g = GridBlock_2D.chebnew(N, N)
    window = app.Window(800, 800, color=(1, 1, 1, 1))
    gen(window, s, g)

    duration = 5
    m = namedtuple('m', 'window, duration')(window, duration)
    glumpyanim(m, dest)


def cmd(gen):
    @wraps(gen)
    def _(name, dest):
        s = import_module("spexy.embed.surface.{}".format(name)).s

        N = 48
        g = GridBlock_2D.chebnew(N, N)
        w = app.Window(800, 800, color=(1, 1, 1, 1))
        gen(w, s, g)

        duration = 5
        m = namedtuple('m', 'window, duration')(w, duration)
        glumpyanim(m, dest)

    return _


def surface_vector_id(w, s, g):
    surface_vector(w, s, g, bases_id)


def surface_vector_sqrt(w, s, g):
    surface_vector(w, s, g, bases_sqrt)


def flat_vector_sqrt(w, s, g):
    flat_vector(w, s, g, bases_sqrt)


def flat_vector_id(w, s, g):
    flat_vector(w, s, g, bases_id)


def surface_glumpy(name, dest, theta):
    s = import_module("spexy.embed.surface.{}".format(name)).s
    theta = int(theta)

    N = 16
    w = app.Window(800, 800, color=(1, 1, 1, 1))

    surface_glpy(w, s, N, theta)

    duration = 5
    m = namedtuple('m', 'window, duration')(w, duration)
    glumpyanim(m, dest)


if __name__ == '__main__':
    import argh

    parser = argh.ArghParser()
    parser.add_commands([
        cmd(surface_vector_id),
        cmd(surface_vector_sqrt),
        cmd(flat_vector_sqrt),
        cmd(flat_vector_id),
        surface_glumpy,
    ])
    parser.dispatch()
