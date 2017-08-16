# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np

from spexy.opengl.tesselation import surface_indices, coordinate_delta
from glumpy import gl, gloo, app

vertex = """
attribute vec2 pos;
attribute vec2 vec;

varying vec2 v_vec;

void main()
{
    v_vec = vec;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

fragment = """
#include "math/constants.glsl"
#include "arrows/arrows.glsl"
#include "antialias/antialias.glsl"

uniform float linewidth;
uniform float antialias;
uniform float rows;
uniform float cols;
uniform vec4 color;

uniform vec2 res;

varying vec2 v_vec;

void main()
{
    float body = min(res.x/cols, res.y/rows) / M_SQRT2;
    vec2 size   = res.xy / vec2(cols,rows);

    vec2 V = v_vec;
    float Vr = sqrt(V.x*V.x + V.y*V.y);
    float Vt = atan(-V.y, V.x);

    vec2 pos = gl_FragCoord.xy;
    vec2 center = (floor(pos/size) + vec2(0.5,0.5)) * size;
    pos -= center;

    pos = vec2(
            cos(Vt)*pos.x - sin(Vt)*pos.y,
            sin(Vt)*pos.x + cos(Vt)*pos.y
    );

    body *= Vr;

    float d = arrow_curved(pos, body, 0.08*body, linewidth, antialias);
    // float d = arrow_stealth(pos, body, 0.15*body, linewidth, antialias);
    // float d = arrow_triangle_90(pos, body, 0.15*body, linewidth, antialias);
    // float d = arrow_triangle_60(pos, body, 0.20*body, linewidth, antialias);
    // float d = arrow_triangle_30(pos, body, 0.25*body, linewidth, antialias);
    // float d = arrow_angle_90(pos, body, 0.15*body, linewidth, antialias);
    // float d = arrow_angle_60(pos, body, 0.20*body, linewidth, antialias);
    // float d = arrow_angle_30(pos, body, 0.25*body, linewidth, antialias);

    gl_FragColor = filled(d, linewidth, antialias, color);
    // gl_FragColor = stroke(d, linewidth, antialias, color);
}
"""


class vectorfield(object):
    def __init__(
            self,
            VT,
            rows=24,
            cols=24,
            linewidth=2,
            antialias=1,
            color=(0, 0, 0, 1),
            deltax=None, deltay=None,
    ):
        u, v = coordinate_delta(deltax, deltay, VT[0].shape)
        x = 2 * u - 1
        y = 2 * v - 1
        X, Y = np.meshgrid(x, y, indexing='ij')

        s_indices = surface_indices(*X.shape)
        s_indices = s_indices.astype(np.uint32).view(gloo.IndexBuffer)

        Vx, Vy = VT
        assert Vx.shape == Vy.shape

        vbuffer = np.empty(X.shape, [
            ('pos', np.float32, 2),
            ('vec', np.float32, 2),
        ]).view(gloo.VertexBuffer)
        vbuffer['pos'][..., 0] = X
        vbuffer['pos'][..., 1] = Y
        vbuffer['vec'][..., 0] = Vx
        vbuffer['vec'][..., 1] = Vy

        program = gloo.Program(vertex, fragment, count=4)
        program.bind(vbuffer)
        program['linewidth'] = linewidth
        program['antialias'] = antialias
        program['rows'] = rows
        program['cols'] = cols
        program['color'] = color

        def clear():
            gl.glClearColor(1, 1, 1, 1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        def on_draw(VT):
            Vx, Vy = VT
            vbuffer['vec'][..., 0] = Vx
            vbuffer['vec'][..., 1] = Vy
            program.draw(gl.GL_TRIANGLES, s_indices)

        def on_resize(width, height):
            program["res"] = width, height

        self.clear = clear
        self.on_draw = on_draw
        self.on_resize = on_resize


if __name__ == '__main__':
    def V0(x, y):
        A = (1 - y ** 2) * (1 - x ** 2)
        return -A * y, A * x


    def Vt(x, y, t=0):
        Vx, Vy = V0(x, y)
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
    rows, cols = 13, 13
    n, m = 512, 512

    u = np.linspace(0, 1, n)
    v = np.linspace(0, 1, m)
    x = 2 * u - 1
    y = 2 * v - 1
    X, Y = np.meshgrid(x, y, indexing='ij')

    vec = vectorfield(
        VT(X, Y, t),
        rows=rows, cols=cols,
        linewidth=2,
        color=(0, 0, 0, 1),
    )

    vec1 = vectorfield(
        VT1(X, Y, t),
        rows=rows, cols=cols,
        linewidth=2,
        color=(1, 0, 0, 1),
    )

    window = app.Window(width=800, height=800)


    @window.event
    def on_draw(_):
        global t
        vec.clear()
        vec.on_draw(VT(X, Y, t))
        vec1.on_draw(VT1(X, Y, t))
        t += dt


    @window.event
    def on_resize(width, height):
        vec.on_resize(width, height)
        vec1.on_resize(width, height)


    app.run()
