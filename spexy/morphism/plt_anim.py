# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import matplotlib.pyplot as plt
from matplotlib import animation

from spexy.matplot import MatplotAnimation
from spexy.morphism.map import IdMap
from spexy.morphism.matrix import smooth_step
from spexy.morphism.plt import plot_grid, plot_jacobian, plot_hodge_star


class MapAnimation(MatplotAnimation):
    def __init__(self, fig, mapping=IdMap(), N=16, M=16, frames=200, interval=20, **kwargs):
        init, draw = plot_grid(fig.gca(), N, M, φ=mapping, **kwargs)

        def animate(i):
            c = 2 * float(i) / float(frames - 1)
            if c > 1: c = 2 - c
            c = smooth_step(.1, .9, c)
            return draw(c)

        ani = animation.FuncAnimation(
            fig, animate,
            init_func=init,
            frames=frames, interval=interval, blit=True,
        )

        MatplotAnimation.__init__(self, ani)


class JacobAnimation(MatplotAnimation):
    def __init__(self, fig, mapping=IdMap(), mode='forward',
                 N=10, M=10, frames=360, interval=20):
        init, draw = plot_jacobian(fig.get_axes(), N, M, φ=mapping, mode=mode)

        def animate(t):
            θ = 360.0 * t / frames
            return draw(θ)

        ani = animation.FuncAnimation(
            fig, animate,
            init_func=init,
            frames=frames, interval=interval, blit=True,
        )

        MatplotAnimation.__init__(self, ani)


class HodgeStarAnimation(MatplotAnimation):
    def __init__(self, fig, mapping=IdMap(),
                 N=10, M=10, frames=360, interval=20):
        init, draw = plot_hodge_star(fig.get_axes(), N, M, φ=mapping)

        def animate(t):
            θ = 360.0 * t / frames
            return draw(θ)

        ani = animation.FuncAnimation(
            fig, animate,
            init_func=init,
            frames=frames, interval=interval, blit=True,
        )

        MatplotAnimation.__init__(self, ani)


def anim(φ, h=5):
    N, M = 16, 16
    fig, ax = plt.subplots(figsize=(h, h))
    return MapAnimation(fig, φ, N=N, M=M, color='black', lw=2)


def anim_jac(φ, mode, h=4):
    fig, ax = plt.subplots(1, 2, figsize=(2 * h, 1 * h))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return JacobAnimation(fig, φ, mode)


def anim_hodge_star(φ, h=4):
    fig, ax = plt.subplots(1, 2, figsize=(2 * h, 1 * h))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return HodgeStarAnimation(fig, φ)
