# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
r"""
Basis functions for primal 0-forms.

.. math::
    \phi_{N,n}^{0}(x)=\alpha_{N,n}(x)

Basis functions for primal 1-forms.

.. math::
    \phi_{N,n}^{1}(x)=\beta_{N,n+\frac{1}{2}}(x)

Basis functions for dual 0-forms.

.. math::
    \tilde{\phi}_{N,n}^{0}(x)=\alpha_{N,n+\frac{1}{2}}(x)

Basis functions for dual 1-forms.

.. math::
    \tilde{\phi}_{N,n}^{1}(x)=\alpha_{N,n}(x)
"""
from spexy.bases import basesimp


class BasesImp(basesimp.BasesImp):
    def module(self):
        return 'spexy.bases.circular'

    def numbers(self):
        N = self.N
        N0 = N
        N1 = N
        N0d = N
        N1d = N
        return (N0, N1), (N0d, N1d)

    def cells_index(self):
        half = self.imp.half
        i0 = lambda n: (n,)
        i1 = lambda n: (n, n + 1)
        id0 = lambda n: (n + half,)
        id1 = lambda n: (n - half, n + half)
        return (i0, i1), (id0, id1)

    def points(self, n):
        return self.imp.points_periodic(self.N, n)

    def bases(self, xp=None):
        imp = self.imp
        N, half = imp.S(self.N), imp.half

        if xp is None:
            xp = imp.points_periodic

        phi0 = lambda n: lambda x: imp.phi(N, x - xp(N, n))
        phi1 = lambda n: lambda x: imp.phi_star(N, x - xp(N, n + half))
        phid0 = lambda n: lambda x: imp.phi(N, x - xp(N, n + half))
        phid1 = lambda n: lambda x: imp.phi_star(N, x - xp(N, n))

        # Gradients
        phi0.grad = lambda n: lambda x: imp.phi_grad(N, x - xp(N, n))
        phid0.grad = lambda n: lambda x: imp.phi_grad(N, x - xp(N, n + half))

        return (phi0, phi1), (phid0, phid1)

    def boundary(self):
        return None, None


def run(N):
    """
    >>> run(1)
    zero-form
    [1]
    one-form
    [1]
    dual zero-form
    [1]
    dual one-form
    [1]

    >>> run(2)
    zero-form
    [1, 0]
    [0, 1]
    one-form
    [1, 0]
    [0, 1]
    dual zero-form
    [1, 0]
    [0, 1]
    dual one-form
    [1, 0]
    [0, 1]

    >>> run(3)
    zero-form
    [1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]
    one-form
    [1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]
    dual zero-form
    [1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]
    dual one-form
    [1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]

    >>> run(4)
    zero-form
    [1, 0, 0, 0]
    [0, 1, 0, 0]
    [0, 0, 1, 0]
    [0, 0, 0, 1]
    one-form
    [1, 0, 0, 0]
    [0, 1, 0, 0]
    [0, 0, 1, 0]
    [0, 0, 0, 1]
    dual zero-form
    [1, 0, 0, 0]
    [0, 1, 0, 0]
    [0, 0, 1, 0]
    [0, 0, 0, 1]
    dual one-form
    [1, 0, 0, 0]
    [0, 1, 0, 0]
    [0, 0, 1, 0]
    [0, 0, 0, 1]

    """
    from spexy.bases.symintegrals import run_integrals
    run_integrals(BasesImp)(N)
