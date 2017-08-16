# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np

from spexy.opengl.tesselation import surface_indices, coordinate_delta
from glumpy import gl, gloo
from spexy.opengl import tesselation

vertex = """

    uniform int drawmesh;
    attribute vec2 a_pos;
    attribute float a_field;
    varying float v_field;

    void main()
    {
        v_field = a_field;
        gl_Position = vec4(a_pos, 0.0, 1.0);
    }
"""

fragment = """
    #include "math/constants.glsl"
    #include "colormaps/colormaps.glsl"

    uniform int drawmesh;
    varying float v_field;

    void main()
    {{
        if (drawmesh == 0) {{
            vec3 color = colormap_{cmap}(v_field, vec3(0, 0, 1), vec3(1, 0, 0));
            gl_FragColor = vec4(color, 1);
        }} else {{
            gl_FragColor = vec4(0, 0, 0, 1);
        }}
    }}
"""


class scalarfield(object):
    """
    x is between -1 and +1
    y is between -1 and +1
    z is between 0 and 1
    """

    def __init__(self, F, mesh=False, deltax=None, deltay=None, cmap='icefire'):
        u, v = coordinate_delta(deltax, deltay, F.shape)
        x = 2 * u - 1
        y = 2 * v - 1
        X, Y = np.meshgrid(x, y, indexing='ij')

        vbuffer = np.empty(X.shape, [
            ('a_pos', np.float32, 2),
            ('a_field', np.float32, 1),
        ]).view(gloo.VertexBuffer)
        vbuffer['a_pos'] = np.rollaxis(np.array([X, Y]), 0, 3)
        vbuffer['a_field'] = F

        program = gloo.Program(vertex, fragment.format(cmap=cmap))
        program.bind(vbuffer)
        program['drawmesh'] = 0

        s_indices = surface_indices(*X.shape)
        s_indices = s_indices.astype(np.uint32).view(gloo.IndexBuffer)

        def on_draw(F):
            vbuffer['a_field'] = F
            program['drawmesh'] = 0
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            program.draw(gl.GL_TRIANGLES, s_indices)

            if mesh:
                program['drawmesh'] = 1
                gl.glLineWidth(1)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
                program.draw(gl.GL_TRIANGLES, s_indices)

        self.on_draw = on_draw


if __name__ == '__main__':
    from glumpy import app


    # def f(x, y, t):
    #     return (np.cos(x) + np.cos(t)) * (np.sin(y) - np.cos(t))

    def f(x, y, t):
        return np.cos(x - t) * np.cos(y - t)


    def F(x, y, t): return (f(7 * x, 7 * y, 3. * t) + 1) / 2


    n = 32
    x = np.linspace(-1, +1, n)
    y = np.linspace(-1, +1, n)
    X, Y = np.meshgrid(x, y, indexing='ij')

    t = 0.0
    field = scalarfield(
        F(X, Y, t),
        mesh=True,
        cmap='icefire',
    )

    window = app.Window(width=800, height=800, color=(1, 1, 1, 1))


    @window.event
    def on_draw(dt):
        global t
        t += dt
        field.on_draw(F(X, Y, t))


    app.run()
