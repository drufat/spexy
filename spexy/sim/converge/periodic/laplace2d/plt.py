from collections import namedtuple

Figs = namedtuple('Figs', ['fig1', 'fig2'])


def plot(sim):
    const = next(sim)
    size = const.size
    var = next(sim)
    L = var.L

    def draw(ax, L0, L1):
        ax.loglog(size, L0, '-o', color='k', label='Primal')
        ax.loglog(size, L1, '-s', color='r', label='Dual')
        ax.grid(True)
        ax.legend()
        ax.set_xlabel(r'$N$')
        ax.set_ylabel(r'$\Vert \Delta f - q \Vert_\infty$')
        ax.set_xticks(
            (4, 8, 16, 32)
        )
        ax.set_xticklabels(
            ('4', '8', '16', '32')
        )
        ax.set_yticks(
            (1e-0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
        )
        ax.set_yticklabels(
            ('1e-0', '1e-2', '1e-4', '1e-6', '1e-8', '1e-10', '1e-12')
        )

    def fig1(ax):
        draw(ax, L[0], L[1])

    def fig2(ax):
        draw(ax, L[2], L[3])

    return Figs(fig1=fig1, fig2=fig2)


if __name__ == '__main__':
    import os
    from spexy.sim.converge.periodic.laplace2d.sim import sim

    s = sim(os.path.join(os.path.dirname(__file__), 'sim.hdf5'))
    # s = sim()
    figs = plot(s)

    import matplotlib.pyplot as plt

    print({f.__name__: f for f in figs})

    for f in figs:
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        f(ax)
    plt.show()
