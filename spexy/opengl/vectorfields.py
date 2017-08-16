# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.

from spexy.opengl.vectorfield import vectorfield


class vectorfields:
    def __init__(
            self,
            VTs,
            colors=(0, 0, 0, 1),
            **kwargs
    ):
        def _(VT, color):
            return vectorfield(VT, color=color, **kwargs)

        fields = [_(v, c) for v, c in zip(VTs, colors)]

        def on_draw(VTs):
            fields[0].clear()
            for f, VT in zip(fields, VTs):
                f.on_draw(VT)

        def on_resize(width, height):
            for f in fields:
                f.on_resize(width, height)

        self.on_draw = on_draw
        self.on_resize = on_resize


if __name__ == '__main__':
    import numpy as np
    from glumpy import app


    def V(x, y):
        A = (1 - y ** 2) * (1 - x ** 2)
        return -A * y, A * x


    def Vt(x, y, t=0):
        Vx, Vy = V(x, y)
        Vx, Vy = (
            + np.cos(t) * Vx - np.sin(t) * Vy,
            + np.sin(t) * Vx + np.cos(t) * Vy
        )
        return Vx, Vy


    def VT(x, y, t):
        Vx, Vy = Vt(x, y, 2 * t)
        Vx, Vy = 3 * Vx, 3 * Vy
        return Vx, Vy


    def VT1(x, y, t):
        Vx, Vy = Vt(x, y, 2 * t)
        Vx, Vy = 3 * Vx, 3 * Vy
        return -Vy, Vx


    t = 0
    dt = 1.5e-2
    rows, cols = 17, 17
    n, m = 512, 512

    x = np.linspace(-1, +1, n)
    y = np.linspace(-1, +1, m)
    X, Y = np.meshgrid(x, y, indexing='ij')

    vecs = vectorfields(
        [VT(X, Y, t), VT1(X, Y, t)],
        rows=rows, cols=cols,
        linewidth=2.,
        colors=[(0, 0, 0, 1), (1, 0, 0, 1)],
    )

    window = app.Window(width=800, height=800)


    @window.event
    def on_draw(_):
        global t
        vecs.on_draw([VT(X, Y, t), VT1(X, Y, t)])
        t += dt


    @window.event
    def on_resize(width, height):
        vecs.on_resize(width, height)


    app.run()
