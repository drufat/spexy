# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.


def cells_lambdify(C0, C1):
    C0 = tuple(zip(C0))
    C1 = tuple(zip(*C1))

    c0 = lambda i: C0[i]
    c1 = lambda i: C1[i]

    n0 = len(C0)
    n1 = len(C1)

    return (c0, c1), (n0, n1)


def cells_012():
    c0 = (0, 1, 2)
    c1 = ((0, 1),
          (1, 2))
    c0d = (0.5, 1.5)
    c1d = ((0.0, 0.5, 1.5),
           (0.5, 1.5, 2.0))
    return (c0, c1), (c0d, c1d)


def cells_345():
    c0 = (3, 4, 5)
    c1 = ((3, 4),
          (4, 5))
    c0d = (3.5, 4.5)
    c1d = ((3.0, 3.5, 4.5),
           (3.5, 4.5, 5.0))
    return (c0, c1), (c0d, c1d)


def cells_3456():
    c0 = (3, 4, 5, 6)
    c1 = ((3, 4, 5),
          (4, 5, 6))
    c0d = (3.5, 4.5, 5.5)
    c1d = ((3.0, 3.5, 4.5, 5.5),
           (3.5, 4.5, 5.5, 6.0))
    return (c0, c1), (c0d, c1d)
