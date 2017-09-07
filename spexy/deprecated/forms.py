# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import itertools


def operators_lambda(n):
    """
    >>> D, H, W, C = operators_lambda(2)
    >>> D(0), D(1), D(2)
    (1, 2, 3)
    >>> H(0), H(1), H(2)
    (2, 1, 0)
    >>> W(0, 0), W(0, 1), W(1, 1), W(0, 2)
    (0, 1, 2, 2)
    >>> C(1, 1), C(1, 2)
    (0, 1)
    """

    def D(k):
        return k + 1

    def H(k):
        return n - k

    def W(k, l):
        return k + l

    def C(k, l):
        assert k == 1
        return l - 1

    return D, H, W, C


def operators_by_degree(n):
    """
    Enumerate all the operators.
    >>> ( operators_by_degree(1) == {
    ... 'D': ((0, 1),),
    ... 'H': ((0, 1), (1, 0)),
    ... 'W': (((0, 0), 0), ((0, 1), 1)),
    ... 'C': (((1, 1), 0),),
    ... })
    True
    >>> ( operators_by_degree(2) == {
    ... 'D': ((0, 1), (1, 2)),
    ... 'H': ((0, 2), (1, 1), (2, 0)),
    ... 'W': (((0, 0), 0), ((0, 1), 1), ((0, 2), 2), ((1, 1), 2)),
    ... 'C': (((1, 1), 0), ((1, 2), 1)),
    ... })
    True
    """
    # enumerate all the possible forms
    D = tuple((k, k + 1) for k in range(n))
    H = tuple((k, n - k) for k in range(n + 1))
    W = tuple(((k, m), k + m) for (k, m) in itertools.product(range(n + 1), range(n + 1)) if (k <= m and k + m <= n))
    C = tuple(((1, k), k - 1) for k in range(1, n + 1))
    return dict(D=D, H=H, W=W, C=C)


def operators(n):
    """
    Return all the operators for dimension n together with their domains and codomains.

    >>> (operators(1) == 
    ... {'D': [('D0', (0, True), (1, True)), 
    ...        ('D0d', (0, False), (1, False))],
    ...  'H': [('H0', (0, True), (1, False)),
    ...        ('H0d', (0, False), (1, True)),
    ...        ('H1', (1, True), (0, False)),
    ...        ('H1d', (1, False), (0, True))],
    ...  'P': [('P0', None, (0, True)),
    ...        ('P0d', None, (0, False)),
    ...        ('P1', None, (1, True)),
    ...        ('P1d', None, (1, False))],
    ...  'R': [('R0', (0, True), None),
    ...        ('R0d', (0, False), None),
    ...        ('R1', (1, True), None),
    ...        ('R1d', (1, False), None)]})
    True
    
    """
    name = lambda n, k, t: '{0}{1}{2}'.format(n, k, 'd' if not t else '')
    # enumerate all the possible discrete forms
    def P(tup): (k, t) = tup; return (name('P', k, t), None, (k, t))

    def R(tup): (k, t) = tup; return (name('R', k, t), (k, t), None)

    def D(tup): (k, t) = tup; return (name('D', k, t), (k, t), (k + 1, t))

    def H(tup): (k, t) = tup; return (name('H', k, t), (k, t), (n - k, not t))

    # Add more operators here - Wedge, Contraction/Flat ?
    forms = tuple(itertools.product(range(n + 1), (True, False)))
    return dict(P=[P(f) for f in forms],
                R=[R(f) for f in forms],
                D=[D(f) for f in forms if f[0] < n],
                H=[H(f) for f in forms])
