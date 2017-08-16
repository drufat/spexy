# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np
from spexy.ops import nat, num
from spexy.helper import eq

np.random.seed(13)


def test_freq():
    for N in N_range:
        assert eq(
            num.freq(N),
            np.array([nat.freq(N, n) for n in range(N)])
        )


Ops = [
    'H', 'Hinv',
    'S', 'Sinv',
    'Q', 'Qinv',
    'G'
]

M = 5
N_range = range(2, 7)


def test_ops():
    for op in Ops:
        for N in N_range:
            f = np.random.rand(N)
            assert eq(
                getattr(num, op)(f),
                getattr(nat, op)(f),
            )


def test_ops_batch():
    for op in Ops:
        for N in N_range:
            f = np.random.rand(M, N)
            assert eq(
                getattr(num, op)(f),
                getattr(nat, op)(f),
            )


def test_slice_():
    for N in N_range:
        f = np.random.rand(M, N)
        for begin, end, step in [
            (0, None, 2),
            (1, None, 2),
            (0, None, 3),
            (1, None, 3),
            (2, None, 3),
            (1, 5, 1),
            (1, 5, 1),
            (1, 5, 2),
            (1, 5, 3),
        ]:
            assert eq(
                nat.slice_(begin, end, step)(f),
                num.slice_(begin, end, step)(f),
            )


def test_roll():
    for N in N_range:
        f = np.random.rand(M, N)
        for n in [
            +1,
            0,
            -1,
            +2, -2, +3,
        ]:
            assert eq(
                nat.roll(n)(f),
                num.roll(n)(f)
            )


def test_weave():
    for N in N_range:
        f1 = np.random.rand(M, N)
        f2 = np.random.rand(M, N)
        assert eq(
            nat.weave(f1, f2),
            num.weave(f1, f2),
        )

        f1 = np.random.rand(M, N + 1)
        f2 = np.random.rand(M, N)
        assert eq(
            nat.weave(f1, f2),
            num.weave(f1, f2),
        )


def test_run():
    assert eq(
        nat.weave([0, 1, 2], [3, 4, 5]),
        np.array([0, 3, 1, 4, 2, 5])
    )
    f1 = np.array([[0, 1, 2, 3, 4],
                   [5, 6, 7, 8, 9]])
    f2 = np.array([[10, 11, 12, 13, 14],
                   [15, 16, 17, 18, 19]])
    assert eq(
        num.weave(f1, f2),
        np.array([[0, 10, 1, 11, 2, 12, 3, 13, 4, 14],
                  [5, 15, 6, 16, 7, 17, 8, 18, 9, 19]])
    )

    assert eq(
        nat.weave(f1, f2),
        np.array([[0, 10, 1, 11, 2, 12, 3, 13, 4, 14],
                  [5, 15, 6, 16, 7, 17, 8, 18, 9, 19]])
    )

    assert eq(
        num.mat(nat.slice_(0, None, 2), 5),
        np.array([[1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1]])
    )
