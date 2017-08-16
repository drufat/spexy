# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import os
from functools import wraps

import matplotlib.pyplot as plt
import sympy as sy

import spexy.morphism.maps
import spexy.matplot
import spexy.morphism.plot as mplt
import spexy.morphism.plt_anim

template = r'''
{mapping}
------------------------------

.. raw:: html

    {video}

.. math::

    {latex}

.. image:: img/mapping.png

The metric is given by

.. math::

    {metric}

Plot the invariants of the metric

.. image:: img/invariants.png

Plot the components of the metric

.. image:: img/metric.png

This is used to check if the map is singular.

.. image:: img/singularity.png

Push forward of the orthogonal frame

.. image:: img/push_forward.png

Pull back of orthogonal frame (parallel to x-axis)


.. image:: img/pull_back_x.png

Pull back of orthogonal frame(parallel to y-axis)

.. image:: img/pull_back_y.png

Pull back of orthogonal frame(sqrt of metric)

.. image:: img/pull_back_sqrt.png

Rotating the push-forward (left frame is orthogonal)

.. raw:: html

    {jacob_forward}

Rotating the pull-back (right frame is orthogonal)

.. raw:: html

    {jacob_backward}

Plotting the flat operator

.. image:: img/flat.png

Plotting the sharp operator

.. image:: img/sharp.png

Plotting the hodge-star operator

.. image:: img/star00.png
.. image:: img/star45.png
.. image:: img/star90.png

Rotating the hodge-star operator

.. raw:: html

    {hodge_star}

'''

stub_template = r'''* :doc:`maps/{mapping}/index`

.. image:: maps/{mapping}/img/mapping.png

'''


def rst(name, filename):
    φ = spexy.morphism.maps.get_map(name)
    x, y = sy.symbols('x, y')
    φ = spexy.morphism.map.Map([x, y], φ(x, y))
    latex = r'\varphi:\quad(x, y) \mapsto \left({}, {}\right)'.format(sy.latex(φ.yy[0]), sy.latex(φ.yy[1]))

    g = sy.simplify(φ.g)
    metric = r'g={}'.format(sy.latex(g))

    fmt = dict(
        mapping=name,
        latex=latex, metric=metric,
        video=spexy.matplot.HTMLAnimation('anim.mp4', inline=False)._repr_html_(),
        jacob_forward=spexy.matplot.HTMLAnimation('forward.mp4', inline=False)._repr_html_(),
        jacob_backward=spexy.matplot.HTMLAnimation('backwardx.mp4', inline=False)._repr_html_(),
        hodge_star=spexy.matplot.HTMLAnimation('star.mp4', inline=False)._repr_html_(),
    )
    txt = template.format(**fmt)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(txt)


def stub(mapping, filename):
    txt = stub_template.format(mapping=mapping)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(txt)


IMG = {
    'img/mapping.png': mplt.mapping,
    'img/invariants.png': mplt.invariants,
    'img/metric.png': mplt.metric,
    'img/singularity.png': mplt.singularity,
    'img/push_forward.png': lambda φ: mplt.jacobian(φ, 'forward'),
    'img/pull_back_x.png': lambda φ: mplt.jacobian(φ, 'backwardx'),
    'img/pull_back_y.png': lambda φ: mplt.jacobian(φ, 'backwardy'),
    'img/pull_back_sqrt.png': mplt.jacobian_sqrt,
    'img/flat.png': lambda φ: mplt.plot_metric(φ, 'flat'),
    'img/sharp.png': lambda φ: mplt.plot_metric(φ, 'sharp'),
    'img/star00.png': lambda φ: mplt.plot_hodge_star(φ, 00),
    'img/star45.png': lambda φ: mplt.plot_hodge_star(φ, 45),
    'img/star90.png': lambda φ: mplt.plot_hodge_star(φ, 90),

}

IMG = {**IMG, **{_.replace('.png', '.pdf'): IMG[_] for _ in IMG}}

MOVIES = {
    'anim.mp4': spexy.morphism.plt_anim.anim,
    'forward.mp4': lambda φ: spexy.morphism.plt_anim.anim_jac(φ, 'forward'),
    'backwardx.mp4': lambda φ: spexy.morphism.plt_anim.anim_jac(φ, 'backwardx'),
    'star.mp4': spexy.morphism.plt_anim.anim_hodge_star,
}

TXT = {
    'stub.txt': stub,
    'index.rst': rst,
}


def plots():
    return [*IMG.keys(), *TXT.keys()]


def movies():
    return [*MOVIES.keys()]


def gen(source, dest):
    mapping = source.stem
    φ = spexy.morphism.maps.get_map(mapping)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    t = dest.split('/{}/'.format(mapping))[-1]
    if t in IMG:
        IMG[t](φ)
        plt.savefig(dest)
    elif t in MOVIES:
        MOVIES[t](φ).save(dest)
    elif t in TXT:
        TXT[t](mapping, dest)
    else:
        raise ValueError


cmds = [
    plots, movies,
    gen,
]

if __name__ == '__main__':
    import argh

    parser = argh.ArghParser()
    parser.add_commands(cmds)
    parser.dispatch()
