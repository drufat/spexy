from collections import namedtuple

Figs = namedtuple('Figs', ['fig0', 'fig1'])


def plot(sim):
    const = next(sim)
    size = const.size
    var = next(sim)

    def fix_axis_labels(ax):
        ax.set_xticks(
            (4, 8, 16, 32, 64)
        )
        ax.set_xticklabels(
            ('4', '8', '16', '32', '64')
        )
        ax.set_yticks(
            (1e-0, 1e-2, 1e-4, 1e-6, 1e-8)
        )
        ax.set_yticklabels(
            ('1e-0', '1e-2', '1e-4', '1e-6', '1e-8')
        )

    def fig0(ax):
        ax.loglog(size, var.L0, '-s', color='k', label='primal')
        ax.loglog(size, var.L0d, '-o', color='r', label='dual')
        ax.grid(True)
        ax.legend()

        ax.set_xlabel(r'$N$')
        ax.set_ylabel(r'$\Vert \Delta f - q \Vert_\infty$')
        fix_axis_labels(ax)

    def fig1(ax):
        ax.loglog(size, var.L1, '-s', color='k', label='primal')
        ax.loglog(size, var.L1d, '-o', color='r', label='dual')
        ax.grid(True)
        ax.legend()

        ax.set_xlabel(r'$N$')
        ax.set_ylabel(r'$\Vert \Delta f - q \Vert_\infty$')
        fix_axis_labels(ax)

    return Figs(fig0=fig0, fig1=fig1)
