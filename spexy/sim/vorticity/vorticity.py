from collections import namedtuple
from importlib._bootstrap_external import SourceFileLoader

import numpy as np
import sympy as sy
from matplotlib import pyplot as plt

from spexy.form import sym, coch
from spexy.helper import to_matrix
from licpy.pixelize import pixelize
from spexy.matplot import grey_save
from spexy.sim.save import save

ConstVort = namedtuple('ConstVort', ['ww'])
VarVort = namedtuple('VarVort', ['vx', 'vy'])


@save(ConstVort, VarVort)
def vort_sim(g, f, n=128):
    x, y = sy.Dummy(), sy.Dummy()
    ω = sym.Form(0, [x, y], [f(x, y)])
    ω = ω.P(g.dual).H

    def Lf(a):
        f = coch.Form(2, g, a)
        f = f.blk
        f = f.H.D.H.D
        f = f.blkinv
        return f.array

    L = to_matrix(Lf, g.N[2])
    Linv = np.linalg.pinv(L)
    rslt = np.dot(Linv, ω.array)
    β = coch.Form(2, ω.grid, rslt)
    v = β.H.D.H

    ((x, y),) = g.blk.refine().verts()
    deltax, deltay = x[:, 0], y[0, :]
    ww = pixelize(n, n, deltax, deltay, ω.blk.Pup.comp[0].array[0])
    vx = pixelize(n, n, deltax, deltay, v.blk.Pup.comp[0].array[0])
    vy = pixelize(n, n, deltax, deltay, v.blk.Pup.comp[1].array[0])

    yield ConstVort(ww)
    yield VarVort(vx, vy)


def vort_save(source, dest, grid):
    m = SourceFileLoader('_', source).load_module()
    if grid == 'primal':
        s = vort_sim(m.g, m.f)
    if grid == 'dual':
        s = vort_sim(m.g.dual, m.f)
    s.write(dest, 1)


def vort_vv(source, dest):
    sim = vort_sim(source)
    ww, = next(sim)
    vx, vy = next(sim)

    from licpy.lic import runlic

    NN = ww.shape[0]
    L = NN // 16
    tex = runlic(vx, vy, L, magnitude=False)
    grey_save(dest, tex)


def vort_ww(source, dest):
    sim = vort_sim(source)
    ww, = next(sim)

    from glumpy import app, gl
    from spexy.opengl.scalarfield import scalarfield

    rescale = lambda z: (z + 1) / 2

    field = scalarfield(rescale(ww))

    window = app.Window(width=ww.shape[0], height=ww.shape[1])
    fbuffer = np.zeros((window.height, window.width, 3), dtype=np.uint8)

    @window.event
    def on_draw(_):
        field.on_draw(rescale(ww))
        gl.glReadPixels(0, 0, window.width, window.height,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, fbuffer)
        plt.imsave(dest, np.flipud(fbuffer))
        app.quit()

    app.run()
