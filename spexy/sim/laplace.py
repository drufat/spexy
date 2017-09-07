from collections import namedtuple
from importlib.machinery import SourceFileLoader

import numpy as np
from matplotlib import pyplot as plt

from spexy.embed.jacob import SJ_sym, SJ
from spexy.form import coch
from spexy.form.coch import Form
from spexy.helper import to_matrix
from licpy.pixelize import pixelize
from spexy.matplot import grey_save
from spexy.opengl.surface import surface
from spexy.sim.metric import bases_sqrt
from spexy.sim.save import save

Const0 = namedtuple('Const0', ['verts'])
Var0 = namedtuple('Var0', ['x', 'y', 'v'])


@save(Const0, Var0)
def sim0(m):
    yield Const0(verts=m.verts)
    yield Var0(x=m.x, y=m.y, v=m.v)


Const1 = namedtuple('Const1', ['verts', 'cn'])
Var1 = namedtuple('Var1', ['x', 'y', 'vx', 'vy'])


@save(Const1, Var1)
def sim1_flat(g, bc0, bc1):
    bc = bc0 + bc1.H.D.H

    if bc.grid != g:
        bc = bc.H
    assert bc.grid == g

    def Lf(a):
        f = Form(1, g, a)
        f = f.blk
        f = f.H.D.H.D + f.D.H.D.H
        f = f.blkinv
        return f.array

    L = to_matrix(Lf, g.N[1])

    Linv = np.linalg.inv(L)
    rslt = -np.dot(Linv, bc.array)
    f = Form(1, g, rslt)
    v = f.blk.Pup

    cn = np.log10(np.linalg.cond(L))
    verts = g.verts()
    vx, vy = [_.array[0] for _ in v.comp]
    ((x, y),) = g.blk.refine().verts()

    yield Const1(verts=verts, cn=cn)
    yield Var1(x=x, y=y, vx=vx, vy=vy)


@save(Const1, Var1)
def sim1_metric(g, s, bc0, bc1):
    S0, J0 = SJ(s, g.blk)(0)
    J, Jinv = bases_sqrt(*J0)
    J = [coch.Form(0, g.blk, [_]).Pup.comp[0].array for _ in J]
    Jinv = [coch.Form(0, g.blk, [_]).Pup.comp[0].array for _ in Jinv]

    bc0 = bc0.blk
    bc1 = bc1.blk

    bc = bc0 + bc1.HH(J, Jinv).D.HH(J, Jinv)
    if bc.grid != g.blk:
        bc = bc.HH(J, Jinv)
    assert bc.grid == g.blk
    bc = bc.blkinv

    def Lf(a):
        f = Form(1, g, a)
        f = f.blk
        f = f.HH(J, Jinv).D.HH(J, Jinv).D + f.D.HH(J, Jinv).D.HH(J, Jinv)
        f = f.blkinv
        return f.array

    L = to_matrix(Lf, g.N[1])

    Linv = np.linalg.inv(L)
    rslt = -np.dot(Linv, bc.array)
    f = Form(1, g, rslt)
    v = f.blk.Pup.sharp(J, Jinv)

    cn = np.log10(np.linalg.cond(L))
    verts = g.verts()
    vx, vy = [_.array[0] for _ in v.comp]
    ((x, y),) = g.blk.refine().verts()

    yield Const1(verts=verts, cn=cn)
    yield Var1(x=x, y=y, vx=vx, vy=vy)


def lap0_save(source, dest):
    s = SourceFileLoader('_', source).load_module()
    sim0(s).write(dest, 1)


def lap0(source, dest):
    s = sim0(source)
    const = next(s)
    var = next(s)
    x, y, v = var.x, var.y, var.v
    vmin = np.min(v)
    vmax = np.max(v)
    if np.allclose(vmin, vmax):
        v[:] = 0.5
    else:
        v = (v - vmin) / (vmax - vmin)
    N, M = 128, 128
    v = pixelize(N, M, x[:, 0], y[0, :], v)
    grey_save(dest, v)


def surface_glpy(s, n, x, y, vx, vy, filename, theta=50.0):
    from glumpy import gl, app
    from licpy.lic import runlic_resample

    N, M = 128, 128
    L = N // 12
    tex = runlic_resample(N, M, x[:, 0], y[0, :], vx, vy, L, magnitude=False)
    tex = tex.T

    u = -np.cos(np.linspace(0, np.pi, n + 1))
    v = -np.cos(np.linspace(0, np.pi, n + 1))
    U, V = np.meshgrid(u, v, indexing='ij')

    sj = SJ_sym(s, U, V)

    deltax = np.diff(u)
    deltay = np.diff(v)

    texture = np.empty(tex.shape + (4,), np.float32)
    texture[:, :, 0] = tex
    texture[:, :, 1] = tex
    texture[:, :, 2] = tex
    texture[:, :, 3] = 1

    surf = surface(
        *sj(0),
        color=(1, 1, 1, 1),
        texture=texture,
        deltax=deltax, deltay=deltay,
    )

    w = app.Window(width=800, height=800)
    w.attach(surf.transform)
    surf.transform.phi = 0.0
    surf.transform.theta = theta

    @w.event
    def on_init():
        surf.on_init()
        surf.on_draw(*sj(0))

    fbuffer = np.zeros((w.height, w.width, 3), dtype=np.uint8)

    @w.event
    def on_draw(_):
        surf.on_draw(*sj(0))
        gl.glReadPixels(0, 0, w.width, w.height,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, fbuffer)
        plt.imsave(filename, np.flipud(fbuffer))

    w.hide()
    w.dispatch_event('on_init')
    w.dispatch_event('on_resize', w._width, w._height)
    w.dispatch_event('on_draw', 0.0)
    w.close()


def lap1_save(source, dest):
    m = SourceFileLoader('_', source).load_module()
    if hasattr(m, 's'):
        s = sim1_metric(m.g, m.s, m.bc0, m.bc1)
    else:
        s = sim1_flat(m.g, m.bc0, m.bc1)
    s.write(dest, 1)


def lap1(source, source_module, dest):
    m = SourceFileLoader('_', source_module).load_module()
    if hasattr(m, 's'):
        s = sim1_metric(source)
    else:
        s = sim1_flat(source)
    const = next(s)
    var = next(s)
    N = 24
    x, y, vx, vy = var.x, var.y, var.vx, var.vy
    vx = pixelize(N, N, x[:, 0], y[0, :], vx)
    vy = pixelize(N, N, x[:, 0], y[0, :], vy)
    xx = pixelize(N, N, x[:, 0], y[0, :], x)
    yy = pixelize(N, N, x[:, 0], y[0, :], y)
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.axis('equal')
    plt.title('cn={:1.3f}'.format(float(const.cn)))
    plt.scatter(*const.verts)
    plt.quiver(xx, yy, vx, vy)
    plt.show()
    plt.savefig(dest)


def lap1_lic(source, source_module, dest):
    m = SourceFileLoader('_', source_module).load_module()
    if hasattr(m, 's'):
        s = sim1_metric(source)
    else:
        s = sim1_flat(source)
    const = next(s)
    var = next(s)
    x, y, vx, vy = var.x, var.y, var.vx, var.vy
    N, M = 128, 128
    # N, M = 256, 256
    L = N // 12
    from licpy.lic import runlic_resample
    tex = runlic_resample(N, M, x[:, 0], y[0, :], vx, vy, L, magnitude=False)
    grey_save(dest, tex)


def lap1_glpy(source, source_module, dest, theta):
    print(f"lap1_glpy('{source}', '{source_module}', '{dest}', '{theta}')")
    theta = int(theta)
    m = SourceFileLoader('_', source_module).load_module()
    s = sim1_metric(source)
    const = next(s)
    var = next(s)
    x, y, vx, vy = var.x, var.y, var.vx, var.vy

    surface_glpy(m.s, 512, x, y, vx, vy, dest, theta)


if __name__ == '__main__':
    import argh

    parser = argh.ArghParser()
    parser.add_commands([
        lap0_save, lap0,
        lap1_save, lap1, lap1_lic, lap1_glpy,
    ])
    parser.dispatch()
