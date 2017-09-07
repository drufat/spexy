# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
# coding: utf-8

# ## Grids

import sympy as sy

from spexy.form import sym
from spexy.grid import Grid_2D

g = Grid_2D.chebnew(8, 8)

x, y = sy.symbols('x, y')
F0, F1, F2 = sym.Form.forms(x, y)


def fields(source, dest):
    import matplotlib.pyplot as plt
    import spexy.plot
    plotf = spexy.plot.plotf
    plt.figure(figsize=(6, 6))
    with open(source, 'r') as f:
        code = f.read()
    exec(code, globals(), locals())
    plt.xlim(g.blk.gx.xmin, g.blk.gx.xmax)
    plt.ylim(g.blk.gy.xmin, g.blk.gy.xmax)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(dest)
