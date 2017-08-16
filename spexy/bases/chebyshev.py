# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
r"""
Basis functions for primal 0-forms.

.. math::
    \psi_{N,n}^{0}(x)=\kappa_{N,n}^{0}(\arccos(-x))

Basis functions for primal 1-forms.

.. math::
    \psi_{N,n}^{1}(x)\mathbf{d}x=
        \kappa_{N,n}^{1}(\arccos(-x))\frac{\mathbf{d}x}{\sqrt{1-x^{2}}}

Basis functions for dual 0-forms.

.. math::
    \tilde{\psi}_{N,n}^{0}(x)=\tilde{\kappa}_{N,n}^{0}(\arccos(-x))

Basis functions for dual 1-forms.

.. math::
    \tilde{\psi}_{N,n}^{1}(x)\mathbf{d}x=\tilde{\kappa}_{N,n}^{1}(\arccos(-x))\frac{\mathbf{d}x}{\sqrt{1-x^{2}}}
"""
from spexy.bases import regular


class BasesImp(regular.BasesImp):
    def points(self, n):
        return self.imp.points_chebyshev_clamped(self.N, n)

    def bases(self, correct=True):
        imp = self.imp
        N, half = imp.S(self.N), imp.half

        def corr0(psi):
            # primal boundary vertex
            if correct:
                return lambda N, n, x: psi(N, n, x) * imp.correction0(N, n)
            return psi

        def corrd1(psi_star):
            # dual boundary edge
            if correct:
                return lambda N, n, x: psi_star(N, n, x) + imp.correctionpsid1(N, n, x)
            return psi_star

        psi0 = lambda n: lambda x: corr0(imp.psi)(N, n, x)
        psi1 = lambda n: lambda x: imp.psi_star(N, n + half, x)
        psid0 = lambda n: lambda x: imp.psi(N, n + half, x)
        psid1 = lambda n: lambda x: corrd1(imp.psi_star)(N, n, x)

        #Gradients
        psi0.grad = lambda n: lambda x: corr0(imp.psi_grad)(N, n, x)
        psid0.grad = lambda n: lambda x: imp.psi_grad(N, n + half, x)

        return (psi0, psi1), (psid0, psid1)

    def boundary(self):
        return None, (-1, +1)


def run_psi():
    """
    >>> from sympy.abc import x

    >>> (psi0, psi1), (psid0, psid1) = BasesImp(1, 'sym').bases()

    >>> psi0(0)(x)
    -x/2 + 1/2
    >>> psi0(1)(x)
    x/2 + 1/2

    >>> psi1(0)(x)
    1/2

    >>> psid0(0)(x)
    1

    >>> psid1(0)(x)
    -x + 1/2
    >>> psid1(1)(x)
    x + 1/2

    >>> (psi0, psi1), (psid0, psid1) = BasesImp(2, 'sym').bases()

    >>> psi0(0)(x)
    x**2/2 - x/2
    >>> psi0(1)(x)
    -x**2 + 1
    >>> psi0(2)(x)
    x**2/2 + x/2

    >>> psi1(0)(x)
    -x + 1/2
    >>> psi1(1)(x)
    x + 1/2

    >>> psid0(0)(x)
    -sqrt(2)*x/2 + 1/2
    >>> psid0(1)(x)
    sqrt(2)*x/2 + 1/2

    >>> from spexy.symbolic import poly_simplify
    >>> p = lambda expr: poly_simplify(expr, x)
    >>> p(psid1(0)(x))
    3*x**2 - 2*x - 1/2
    >>> p(psid1(1)(x))
    x**2*(-3*sqrt(2) + 3) - 1/2 + sqrt(2)
    >>> p(psid1(2)(x))
    3*x**2 + 2*x - 1/2

    """
    pass


def run(N):
    """
    >>> run(1)
    zero-form
    [1, 0]
    [0, 1]
    one-form
    [1]
    dual zero-form
    [1]
    dual one-form
    [1, 0]
    [0, 1]

    >>> run(2)
    zero-form
    [1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]
    one-form
    [1, 0]
    [0, 1]
    dual zero-form
    [1, 0]
    [0, 1]
    dual one-form
    [1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]

    >>> run(3)
    zero-form
    [1, 0, 0, 0]
    [0, 1, 0, 0]
    [0, 0, 1, 0]
    [0, 0, 0, 1]
    one-form
    [1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]
    dual zero-form
    [1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]
    dual one-form
    [1, 0, 0, 0]
    [0, 1, 0, 0]
    [0, 0, 1, 0]
    [0, 0, 0, 1]
    """
    from spexy.bases.symintegrals import run_integrals
    run_integrals(BasesImp)(N)
