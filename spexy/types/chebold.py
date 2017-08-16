# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
"""
>>> from spexy.grid import Grid_1D
>>> from spexy.grid.grid import hodge_star_matrix, upsample_matrix
>>> g = Grid_1D.chebold(2)
>>> H0, H1, H0d, H1d = hodge_star_matrix(g)
>>> H0d
array([[ 0.854,  0.146],
       [ 0.146,  0.854]])
>>> H1
array([[ 1.207, -0.207],
       [-0.207,  1.207]])
>>> H0
array([[ 0.233,  0.077, -0.017],
       [ 0.118,  1.179,  0.118],
       [-0.017,  0.077,  0.233]])
>>> H1d
array([[ 4.5  , -0.328,  0.5  ],
       [-0.5  ,  0.914, -0.5  ],
       [ 0.5  , -0.328,  4.5  ]])
"""

xmin, xmax = -1, +1

from spexy.types.chebyshev import (
    derivative,
    hodge_star,
    upsample,
    downsample,
    gradient,
)

derivative
hodge_star
upsample
downsample
gradient
