# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import matplotlib.pyplot as plt
import sympy as sy

import spexy.morphism.plt
import spexy.morphism.plt_anim
from spexy.morphism.plt_anim import anim_jac


def mapping(φ):
    N, M = 16, 16
    # N, M = 32, 32
    # N, M = 64, 64
    num = 4
    h = 4
    fig, ax = plt.subplots(1, num, figsize=(num * h, h))
    for i in range(num):
        c = float(i) / float(num - 1)
        init, draw = spexy.morphism.plt.plot_grid(ax[i], N, M, 100, φ, color='black', lw=2)
        draw(c)
    plt.subplots_adjust(wspace=0.05)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)


def invariants(φ):
    I, II, III = φ.metric_invariants()

    h = 5
    fig, ax = plt.subplots(1, 3, figsize=(3 * h, h))
    spexy.morphism.plt.plot_st(ax[0], φ.xx, I, title='$I_g-2$', center=2)
    spexy.morphism.plt.plot_st(ax[1], φ.xx, II, title='$II_g-1$', center=1)
    spexy.morphism.plt.plot_st(ax[2], φ.xx, III, title='$III_g-1$', center=1)
    plt.subplots_adjust(wspace=0.05)


def metric(φ):
    g = φ.metric()

    h = 5
    fig, ax = plt.subplots(2, 2, figsize=(2 * h, 2 * h))
    spexy.morphism.plt.plot_st(ax[0][0], φ.xx, g[0, 0] - 1, title='$g_{xx}-1$', center=0)
    spexy.morphism.plt.plot_st(ax[0][1], φ.xx, g[0, 1], title='$g_{xy}$', center=0)
    spexy.morphism.plt.plot_st(ax[1][0], φ.xx, g[1, 0], title='$g_{xy}$', center=0)
    spexy.morphism.plt.plot_st(ax[1][1], φ.xx, g[1, 1] - 1, title='$g_{yy}-1$', center=0)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)


def singularity(φ):
    plt.figure()
    plt.title('$\log_{10}(\det(g))$')
    spexy.morphism.plt.plot_st(plt.gca(), φ.xx, sy.log(abs(φ.J.det()), 10), cmap=plt.cm.bone, colorbar=True)


def jacobian(φ, mode='forward'):
    h = 7
    n = 10
    m = 10

    fig, ax = plt.subplots(1, 2, figsize=(2 * h, h))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    init, draw = spexy.morphism.plt.plot_jacobian(ax, n=n, m=m, φ=φ, mode=mode)
    draw(0)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)


def jacobian_sqrt(φ):
    h = 7
    n = 10
    m = 10

    fig, ax = plt.subplots(1, 2, figsize=(2 * h, h))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    spexy.morphism.plt.plot_jacobian_sqrt(ax, n=n, m=m, φ=φ)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)


def plot_metric(φ, mode='flat'):
    h = 7
    n = 10
    m = 10

    fig, ax = plt.subplots(1, 2, figsize=(2 * h, h))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    init, draw = spexy.morphism.plt.plot_metric(ax, n, m, φ=φ, mode=mode)
    draw(0)
    plt.suptitle(r'$\{}$'.format(mode), size=30)
    plt.subplots_adjust(left=0, right=1, bottom=0)


def plot_hodge_star(φ, θ=0):
    h = 7
    n = 10
    m = 10

    fig, ax = plt.subplots(1, 2, figsize=(2 * h, h))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    init, draw = spexy.morphism.plt.plot_hodge_star(ax, n, m, φ=φ)
    draw(θ)
    plt.suptitle(r'$\star$', size=30)
    plt.subplots_adjust(left=0, right=1, bottom=0)


if __name__ == '__main__':
    from spexy.morphism.maps.gerritsma import φ

    anim = anim_jac(φ, 'forward')

    mapping(φ)
    invariants(φ)
    metric(φ)
    plot_metric(φ, 'flat')
    plt.show()
