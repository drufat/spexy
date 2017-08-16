# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np

import spexy.spectral as sp

xmin, xmax = 0, np.pi

from .chebyshev import D0, D1, D0d, D1d


@sp.batch
def H0d(f):
    r"""

    .. math::
        \tilde{\mathbf{H}}^{0}=
            \mathbf{M}_{1}^{\dagger}
            \mathbf{I}^{-\frac{h}{2},\frac{h}{2}}
            \mathbf{M}_{1}^{+}
    """
    f = sp.mirror1(f, +1)
    f = sp.I_space(-0.5, +0.5)(f)
    f = sp.unmirror1(f)
    return f


@sp.batch
def H1(f):
    r"""

    .. math::
        \mathbf{H}^{1}=
            \mathbf{M}_{1}^{\dagger}
            {\mathbf{I}^{-\frac{h}{2},\frac{h}{2}}}^{-1}
            \mathbf{M}_{1}^{+}
    """
    f = sp.mirror1(f, +1)
    f = sp.I_space_inv(-0.5, +0.5)(f)
    f = sp.unmirror1(f)
    return f


@sp.batch
def H0(f):
    r"""

    .. math::
        \mathbf{H}^{0}=
            \mathbf{A}
            \mathbf{M}_{0}^{\dagger}
            \mathbf{I}^{-\frac{h}{2},\frac{h}{2}}
            \mathbf{M}_{0}^{+}
    """
    f = sp.mirror0(f, +1)
    f = sp.I_space(-0.5, +0.5)(f)
    f = sp.unmirror0(f)
    f = f * sp.A_diag(f.shape[0])
    return f


@sp.batch
def H1d(f):
    r"""

    .. math::
        \tilde{\mathbf{H}}^{1}=
            \mathbf{M}_{1}^{\dagger}
            {\mathbf{I}^{-\frac{h}{2},\frac{h}{2}}}^{-1}
            \mathbf{M}_{1}^{+}
            \mathbf{A}^{-1}
    """
    f = f / sp.A_diag(f.shape[0])
    f = sp.mirror0(f, +1)
    f = sp.I_space_inv(-0.5, +0.5)(f)
    f = sp.unmirror0(f)
    return f


derivative = (
    (D0, D1),
    (D0d, D1d),
)

hodge_star = (
    (H0, H1),
    (H0d, H1d),
)

upsample = (
    (None, None),
    (None, None),
)

downsample = (
    (None, None),
    (None, None),
)

gradient = (
    (None, None),
    (None, None),
)
