# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.

import numpy as np
import sympy as sy

from spexy.opengl import checker
from spexy.opengl.tesselation import surface_indices, boundary_indices, coordinate_delta
from glumpy import gl, gloo
from glumpy.transforms import Trackball

vert = """
    attribute vec2 a_tex;
    attribute vec3 a_pos;
    attribute vec3 a_eu;
    attribute vec3 a_ev;

    varying vec3 v_pos;
    varying vec2 v_tex;
    varying vec3 v_ex;
    varying vec3 v_ey;

    void main()
    {
        v_tex = a_tex;
        v_pos = a_pos;
        v_ex = a_eu;
        v_ey = a_ev;
        gl_Position = <transform(vec4(a_pos, 0.6))>;
    }
"""

fragment = """
    #include "math/constants.glsl"

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 normal;
    uniform vec4 color;
    uniform sampler2D texture;

    uniform vec3 light_color[3];
    uniform vec3 light_pos[3];

    varying vec2 v_tex;
    varying vec3 v_pos;
    varying vec3 v_ex;
    varying vec3 v_ey;

    float lighting(vec3 v_normal, vec3 light_pos)
    {
        // Calculate normal in world coordinates
        vec3 n = normalize(normal * vec4(v_normal,1.0)).xyz;

        // Calculate the location of this fragment (pixel) in world coordinates
        vec3 pos = vec3(view * model * vec4(v_pos, 1));

        // Calculate the vector from this pixels program to the light source
        vec3 surface_to_light = light_pos - pos;

        // Calculate the cosine of the angle of incidence (brightness)
        float brightness = dot(n, surface_to_light) / (length(surface_to_light) * length(n));
        brightness = max(min(brightness,1.0),0.0);
        return brightness;
    }

    void main()
    {
        vec3 v_normal = normalize(cross(v_ex, v_ey));
        vec4 l1 = vec4(light_color[0] * lighting(v_normal, light_pos[0]), 1);
        vec4 l2 = vec4(light_color[1] * lighting(v_normal, light_pos[1]), 1);
        vec4 l3 = vec4(light_color[2] * lighting(v_normal, light_pos[2]), 1);

        <surfacecolor>
        //float c = 0.6 + 0.4*texture2D(texture, pos).r;
        //vec4 surfacecolor = vec4(c,c,c,1);

        // Map value to rgb color
        gl_FragColor = color * surfacecolor * (0.4 + 0.6*(l1+l2+l3));
    }
"""

LIGHT_POS = [
    # [+2, +2, +2],
    # [+2, +2, +2],
    # [+2, +2, +2],
    [+3, +0, +5],
    [+2, +2, +5],
    [+0, +3, +5],
]
LIGHT_COLOR = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
]


class surface(object):
    def __init__(
            self,
            S, J,
            mesh=False,
            texture=None,
            color=(1, 1, 1, 1),
            bcolor=(0, 0, 0, 1),
            deltax=None, deltay=None,
            light_pos=LIGHT_POS,
            light_color=LIGHT_COLOR,
    ):
        u, v = coordinate_delta(deltax, deltay, S[0].shape)
        U, V = np.meshgrid(u, v, indexing='ij')
        frag = fragment
        if texture is None:
            texture = checker.checkerboard(np.diff(u), np.diff(v), 1024, 1024)
            frag = frag.replace(
                '<surfacecolor>',
                """
                float c = 0.6 + 0.4*texture2D(texture, v_tex).r;
                vec4 surfacecolor = vec4(c,c,c,1);
                """
            )
        else:
            frag = frag.replace(
                '<surfacecolor>',
                """
                vec4 surfacecolor = 0.2 + 0.8*texture2D(texture, v_tex);
                """
            )

        s_indices = surface_indices(*U.shape)
        s_indices = s_indices.astype(np.uint32).view(gloo.IndexBuffer)
        b_indices = boundary_indices(*U.shape)
        b_indices = b_indices.astype(np.uint32).view(gloo.IndexBuffer)

        prog = gloo.Program(vert, frag)
        transform = Trackball()

        X, Y, Z = S
        Jxu, Jxv, Jyu, Jyv, Jzu, Jzv = J
        [
            X, Y, Z,
            Jxu, Jyu, Jzu,
            Jxv, Jyv, Jzv,
        ] = np.broadcast_arrays(
            X, Y, Z,
            Jxu, Jyu, Jzu,
            Jxv, Jyv, Jzv,
        )

        vbuffer = np.empty(Z.shape, [
            ('a_tex', np.float32, 2),
            ('a_pos', np.float32, 3),
            ('a_eu', np.float32, 3),
            ('a_ev', np.float32, 3),
        ]).view(gloo.VertexBuffer)
        vbuffer['a_tex'] = np.rollaxis(np.array([U, V]), 0, 3)
        vbuffer['a_pos'] = np.rollaxis(np.array([X, Y, Z]), 0, 3)
        vbuffer['a_eu'] = np.rollaxis(np.array([Jxu, Jyu, Jzu]), 0, 3)
        vbuffer['a_ev'] = np.rollaxis(np.array([Jxv, Jyv, Jzv]), 0, 3)

        prog.bind(vbuffer)
        prog['texture'] = texture
        prog['transform'] = transform
        prog["light_pos[0]"] = light_pos[0]
        prog["light_pos[1]"] = light_pos[1]
        prog["light_pos[2]"] = light_pos[2]
        prog["light_color[0]"] = light_color[0]
        prog["light_color[1]"] = light_color[1]
        prog["light_color[2]"] = light_color[2]

        def on_init():
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glPolygonOffset(1, 1)
            gl.glEnable(gl.GL_LINE_SMOOTH)
            gl.glLineWidth(3)

        def on_draw(S, J):
            X, Y, Z = S
            Jxu, Jxv, Jyu, Jyv, Jzu, Jzv = J
            [
                X, Y, Z,
                Jxu, Jyu, Jzu,
                Jxv, Jyv, Jzv,
            ] = np.broadcast_arrays(
                X, Y, Z,
                Jxu, Jyu, Jzu,
                Jxv, Jyv, Jzv,
            )
            vbuffer['a_pos'] = np.rollaxis(np.array([X, Y, Z]), 0, 3)
            vbuffer['a_eu'] = np.rollaxis(np.array([Jxu, Jyu, Jzu]), 0, 3)
            vbuffer['a_ev'] = np.rollaxis(np.array([Jxv, Jyv, Jzv]), 0, 3)

            gl.glClearColor(1., 1., 1., 1, )
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glDisable(gl.GL_BLEND)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
            prog['color'] = color
            prog.draw(gl.GL_TRIANGLES, s_indices)

            if mesh:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
                gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
                gl.glEnable(gl.GL_BLEND)
                gl.glDepthMask(gl.GL_FALSE)
                prog['color'] = bcolor
                prog.draw(gl.GL_TRIANGLES, s_indices)
                gl.glDisable(gl.GL_LINE_SMOOTH)
                gl.glDepthMask(gl.GL_TRUE)

            gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
            gl.glEnable(gl.GL_BLEND)
            gl.glDepthMask(gl.GL_FALSE)
            prog["color"] = bcolor
            prog.draw(gl.GL_LINE_LOOP, b_indices)
            gl.glDepthMask(gl.GL_TRUE)

            view = prog['transform']['view'].reshape(4, 4)
            prog['view'] = view
            model = prog['transform']['model'].reshape(4, 4)
            prog['model'] = model
            prog['normal'] = np.array(np.matrix(np.dot(view, model)).I.T)

        self.on_init = on_init
        self.on_draw = on_draw
        self.transform = transform


if __name__ == '__main__':
    from glumpy import app
    from spexy.embed.jacob import SJ_sym


    def func(u, v, t=0):
        x = 2 * u - 1
        y = 2 * v - 1
        z = (-x ** 2 + y ** 2) * sy.cos(t)
        return x, y, z


    def Func(x, y, t):
        return func(x, y, t)


    np.random.seed(13)

    n, m = 16, 16
    # lenu = np.ones(n - 1)
    # lenv = np.ones(m - 1)
    lenu = np.random.rand(n - 1) + .2
    lenv = np.random.rand(m - 1) + .2
    u = np.r_[[0], np.cumsum(lenu)] / sum(lenu)
    v = np.r_[[0], np.cumsum(lenv)] / sum(lenv)
    U, V = np.meshgrid(u, v, indexing='ij')

    sj = SJ_sym(Func, U, V)

    t = 0.0
    surf = surface(
        *sj(t),
        mesh=True,
        zmin=-1, zmax=+1,
        color=(1, 1, 1, 1),
        deltax=lenu, deltay=lenv,
    )

    window = app.Window(800, 800, color=(1, 1, 1, 1))
    window.attach(surf.transform)
    _ = surf.transform
    _.phi = 0
    _.theta = 25
    _.zoom = 42
    _.distance = 6.7


    @window.event
    def on_init():
        surf.on_init()


    @window.event
    def on_draw(dt):
        global t
        surf.on_draw(*sj(t))
        t += dt


    app.run()
