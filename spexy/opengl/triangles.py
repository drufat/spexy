# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np

from glumpy import app, gloo, gl

n = 20
x = np.linspace(-1.0, 1.0, n)
y = np.linspace(-1.0, 1.0, n)
X, Y = np.meshgrid(x, y)

vertices = np.c_[X.reshape(-1), Y.reshape(-1)]

N, M = X.shape
i = np.arange(N * M).reshape(N, M)
i0 = i[:-1, :-1].reshape(-1)
i1 = i[:-1, 1:].reshape(-1)
i2 = i[1:, 1:].reshape(-1)
i3 = i[1:, :-1].reshape(-1)
indices = np.c_[i0, i1, i2, i0, i2, i3].reshape(-1)

vertex = '''
    attribute vec2 a_position;
    void main()
    {
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
'''

fragment = '''
    void main()
    {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
'''

if __name__ == '__main__':
    program = gloo.Program(vertex, fragment)
    program['a_position'] = vertices

    window = app.Window(width=512, height=512, color=(1, 1, 1, 1))


    @window.event
    def on_draw(_):
        window.clear()
        gl.glLineWidth(2)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        program.draw(gl.GL_TRIANGLES, indices.astype(np.uint32).view(gloo.IndexBuffer))


    app.run()
