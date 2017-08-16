# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy

import spexy.continuous as cont
from spexy.helper import load_source

x, y = sy.symbols('x, y')

template = r'''
{name}
..................................

.. math::

    \mathbf{{v}}(x, y) & = & {v} \\
    \omega (x,y) & = & {omega} \\
    p(x,y) & = & {p} \\
    \dot{{\mathbf{{v}} }}(x,y) & = & {vdot}


'''


def img(src, dest):
    m = load_source('m', src)

    plt.figure(figsize=(8, 8))

    scale = [
        -np.pi,
        +np.pi,
    ]
    axes = [
        plt.subplot(221, aspect='equal'),
        plt.subplot(222, aspect='equal'),
        plt.subplot(223, aspect='equal'),
        plt.subplot(224, aspect='equal'),
    ]
    for ax in axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(scale)
        ax.set_ylim(scale)

    n = 20
    X, Y = np.meshgrid(
        np.linspace(*scale, n),
        np.linspace(*scale, n)
    )
    u, v = sy.lambdify((x, y), m.V(x, y), 'numpy')(X, Y)
    axes[0].quiver(X, Y, u + 0 * X, v + 0 * X)
    axes[0].set_title(r'$\mathbf{v}(x,y)$')

    vdot = sy.simplify(cont.v_dot(m.V(x, y), m.p(x, y)))
    udot, vdot = sy.lambdify((x, y), [vdot[0], vdot[1]], 'numpy')(X, Y)
    udot = udot + 0 * X
    vdot = vdot + 0 * X
    axes[2].quiver(X, Y, udot, vdot)
    axes[2].set_title(r'$\mathbf{\dot{v}}(x,y)$')

    omega = sy.simplify(cont.vort(m.V(x, y)))

    n = 200
    X, Y = np.meshgrid(
        np.linspace(*scale, n),
        np.linspace(*scale, n))
    Omega = sy.lambdify((x, y), omega, 'numpy')(X, Y) + 0 * X
    axes[1].contourf(X, Y, Omega)
    axes[1].set_title(r'$\omega(x,y)$')

    P = sy.lambdify((x, y), m.p(x, y), 'numpy')(X, Y) + 0 * X
    axes[3].contourf(X, Y, P)
    axes[3].set_title(r'$p(x,y)$')

    plt.savefig(dest)


def math(src, dest):
    m = load_source('m', src)

    v, p = m.V(x, y), m.p(x, y)
    # pdyn = (v[0] ** 2 + v[1] ** 2) / sy.S(2)
    vdot = cont.v_dot(v, p)
    vort = cont.vort(v)
    vdot = sy.simplify(vdot)
    vort = sy.simplify(vort)

    assert (
        sy.expand_trig(sy.simplify(
            vdot + cont.adv(v) + cont.grad(p)
        )) == sy.Matrix([[0], [0]])
    )

    # print(sy.mathematica_code(
    #     sy.expand_trig(sy.simplify(
    #         cont.div(cont.adv(v) + cont.grad(p))
    #     ))
    # ))

    rst = template.format(
        name=src.stem,
        v=sy.latex(v),
        p=sy.latex(p),
        omega=sy.latex(vort),
        vdot=sy.latex(vdot),
    )
    with open(dest, 'w') as f:
        f.write(rst)


if __name__ == '__main__':
    import argh

    parser = argh.ArghParser()
    parser.add_commands([
        img,
        math,
    ])
    parser.dispatch()
