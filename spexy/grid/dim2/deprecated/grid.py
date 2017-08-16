# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np

from spexy.grid.grid import (unwrap, wrap)
from spexy.strides import (countshape, index_kij_vec)


def flatten_cells_vec(cells, numbers):
    """
    >>> import spexy.grid.dim2.grid_block as gb
    >>> g = gb.GridBlock_2D.chebyshev(2, 2)
    >>> cells = flatten_cells_vec(g.cells, g.N)
    >>> tuple(countshape(g.N[k]) for k in range(g.dimension+1))
    (9, 12, 4)
    >>> a = np.array
    >>> g.cells[0][0](a([0, 1]), a([0, 0]))
    (array([-1., -0.]), array([-1., -1.]))
    >>> cells[0](a([0, 1]))
    (array([-1., -1.]), array([-1., -0.]))
    >>> g.cells[1][0](a([0, 1]), a([0, 0]))
    (array([-1., -0.]), array([-0.,  1.]), array([-1., -1.]))
    >>> cells[1](a([0, 1]))
    (array([-1., -1.]), array([-1., -0.]), array([-0., -0.]), array([-1., -0.]))
    >>> g.cells[1][1](a([0, 1]), a([0, 0]))
    (array([-1., -0.]), array([-1., -1.]), array([-0., -0.]))
    >>> cells[1](a([6, 7]))
    (array([-1., -1.]), array([-1., -0.]), array([-1., -1.]), array([-0.,  1.]))
    """

    def _(c0, c1, c2, n0, n1, n2):
        vidx, vidxinv = index_kij_vec(n0)

        def vert(m):
            # ((x, y),)
            X = np.empty(m.shape)
            Y = np.empty(m.shape)

            k, i, j = vidx(m)
            assert (k == 0).all()

            x, y = c0[0](i, j)
            X[k == 0] = x
            Y[k == 0] = y

            return X, Y

        eidx, eidxinv = index_kij_vec(n1)

        def edge(m):
            # ((x0, y0), (x1, y1))
            X0 = np.empty(m.shape)
            Y0 = np.empty(m.shape)
            X1 = np.empty(m.shape)
            Y1 = np.empty(m.shape)

            k, i, j = eidx(m)
            assert np.logical_xor(k == 0, k == 1).all()

            x0, x1, y = c1[0](i[k == 0], j[k == 0])
            X0[k == 0] = x0
            X1[k == 0] = x1
            Y0[k == 0] = y
            Y1[k == 0] = y

            x, y0, y1 = c1[1](i[k == 1], j[k == 1])
            X0[k == 1] = x
            X1[k == 1] = x
            Y0[k == 1] = y0
            Y1[k == 1] = y1

            return X0, Y0, X1, Y1

        fidx, fidxinv = index_kij_vec(n2)

        def face(m):
            # ((x0, y0), (x1, y0), (x1, y1), (x0, y1),)
            X0 = np.empty(m.shape)
            Y0 = np.empty(m.shape)
            X1 = np.empty(m.shape)
            Y1 = np.empty(m.shape)

            k, i, j = fidx(m)
            assert (k == 0).all()

            x0, x1, y0, y1 = c2[0](i[k == 0], j[k == 0])
            X0[k == 0] = x0
            X1[k == 0] = x1
            Y0[k == 0] = y0
            Y1[k == 0] = y1

            return X0, Y0, X1, Y0, X1, Y1, X0, Y1

        return vert, edge, face

    return _(*cells, *numbers)
