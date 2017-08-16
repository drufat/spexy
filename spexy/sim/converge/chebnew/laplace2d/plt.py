from collections import namedtuple

Figs = namedtuple('Figs', ['fig0', 'fig1'])


def plot(sim):
    const = next(sim)
    size = const.size
    var = next(sim)

    def draw(ax, L0, L1):
        ax.loglog(size, L0, '-o', color='k', label='primal')
        ax.loglog(size, L1, '-s', color='r', label='dual')
        ax.grid(True)
        ax.legend()
        ax.set_xlabel(r'$N$')
        ax.set_ylabel(r'$\Vert \Delta f - q \Vert_\infty$')
        ax.set_xticks(
            (5, 10, 20, 40)
        )
        ax.set_xticklabels(
            (5, 10, 20, 40)
        )
        ax.set_yticks(
            (1e-0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
        )
        ax.set_yticklabels(
            ('1e-0', '1e-2', '1e-4', '1e-6', '1e-8', '1e-10', '1e-12')
        )

    def fig0(ax):
        draw(ax, var.L0, var.L0d)

    def fig1(ax):
        draw(ax, var.L1, var.L1d)

    return Figs(fig0=fig0, fig1=fig1)


if __name__ == '__main__':
    import os
    from spexy.sim.converge.chebold.laplace2d.sim import sim

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
