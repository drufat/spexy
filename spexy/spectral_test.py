# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np

from spexy.helper import to_matrix, eq
from spexy.spectral import (H, Hinv, S, Sinv, I, Iinv, fourier_K, fourier_K_inv)

np.random.seed(1)


def test_transforms():
    x = np.random.random(11)
    eq(x, Sinv(S(x)))
    eq(Hinv(x), S(Hinv(Sinv(x))))
    eq(H(S(x)), S(H(x)))
    eq(x, Iinv(I(x, -1, 1), -1, 1))
    eq(x, I(Iinv(x, -1, 1), -1, 1))


def test_H_S_I():
    f = np.sin
    fp = np.cos

    x = np.linspace(0, 2 * np.pi, 13)[:-1]
    N, = x.shape
    h = 2 * np.pi / N

    eq(H(fp(x)), f(x + h / 2) - f(x - h / 2))
    eq(fp(x), Hinv(f(x + h / 2) - f(x - h / 2)))

    eq(S(f(x)), f(x + h / 2))
    eq(Sinv(f(x)), f(x - h / 2))

    eq(I(fp(x), 0, 1), f(x + h) - f(x))
    eq(I(fp(x), -1 / 2, 1 / 2), f(x + h / 2) - f(x - h / 2))
    eq(I(fp(x), -1, 0), f(x) - f(x - h))
    eq(I(fp(x), -1, 1), f(x + h) - f(x - h))
    eq(fp(x), Iinv(f(x + h) - f(x), 0, 1))
    eq(fp(x), Iinv(f(x + h / 2) - f(x - h / 2), -1 / 2, 1 / 2))
    eq(fp(x), Iinv(f(x) - f(x - h), -1, 0))


def test_linearity():
    for N in range(4, 10):
        h = 2 * np.pi / N
        fk = lambda x: fourier_K(x, 0, h / 2)
        fk_inv = lambda x: fourier_K_inv(x, 0, h / 2)
        for i in range(10):
            a = np.random.rand(N)
            b = np.random.rand(N)
            c = fk_inv(a + i * b)
            d = fk_inv(a) + i * fk_inv(b)
            eq(c, d)
            c = fk(a + i * b)
            d = fk(a) + i * fk(b)
            eq(c, d)


def test_fourier_K_inv():
    for N in range(4, 10):
        h = 2 * np.pi / N
        fk = lambda x: fourier_K(x, 0, h / 2)
        fk_inv = lambda x: fourier_K_inv(x, 0, h / 2)
        a = np.random.rand(N)
        b = fk_inv(fk(a))
        c = fk(fk_inv(a))
        eq(a, b)
        eq(a, c)
        K = to_matrix(fk, N)
        Kinv = to_matrix(fk_inv, N)
        eq(K.dot(a), fk(a))
        eq(Kinv.dot(a), fk_inv(a))
        eq(np.linalg.matrix_rank(Kinv, 1e-5), N)
        eq(np.linalg.inv(K), Kinv)
        eq(np.linalg.inv(Kinv), K)
